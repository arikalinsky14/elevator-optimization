import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import math

# Hyperparameters
discount_factor = 0.99
learning_rate = 0.001
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
replay_buffer_size = 2000

# Build the DQN model
def build_dqn(state_size, action_size):
    model = Sequential([
        Dense(64, input_dim=state_size, activation='relu'),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Function to choose an action based on epsilon-greedy policy
def choose_action(state, model, action_size):
    if np.random.rand() <= epsilon:
        return random.randint(0, action_size - 1)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Replay buffer class
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def size(self):
        return len(self.buffer)

# Train the DQN
def train_dqn(model, target_model, replay_buffer):
    if replay_buffer.size() < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states)
    next_states = np.array(next_states)
    q_values = model.predict(states)
    target_q_values = target_model.predict(next_states)

    for i in range(batch_size):
        if dones[i]:
            q_values[i][actions[i]] = rewards[i]
        else:
            q_values[i][actions[i]] = rewards[i] + discount_factor * np.max(target_q_values[i])

    model.fit(states, q_values, verbose=0)

# Main simulation function
def main():
    floors = 10
    carCapacity = 6
    totalTime = 10000

    state_size = floors * 2 + 3  # Flattened building matrix + elevator state
    action_size = 3  # Up, Down, Stop

    dqn_model = build_dqn(state_size, action_size)
    target_model = build_dqn(state_size, action_size)
    target_model.set_weights(dqn_model.get_weights())

    replay_buffer = ReplayBuffer(replay_buffer_size)

    global epsilon
    total_rewards = []

    for time in range(totalTime):
        building = np.zeros((3, floors))
        directionFloorOccupants = [0, 0, 0]  # Direction, Floor, Occupants

        # Flatten state
        state = np.concatenate((building.flatten(), directionFloorOccupants)).reshape(1, -1)

        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, dqn_model, action_size)

            # Simulate environment dynamics
            next_state, reward, done = simulate_step(building, directionFloorOccupants, action, floors, carCapacity)
            total_reward += reward

            # Add experience to replay buffer
            replay_buffer.add((state, action, reward, next_state, done))

            # Train the DQN
            train_dqn(dqn_model, target_model, replay_buffer)

            state = next_state

        total_rewards.append(total_reward)
        
        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network weights periodically
        if time % 100 == 0:
            target_model.set_weights(dqn_model.get_weights())

    print("Training completed!")

def simulate_step(building, directionFloorOccupants, action, floors, carCapacity):
    """Simulate the elevator dynamics and return next_state, reward, done."""
    reward = 0
    done = False

    # Track waiting times squared
    waiting_times_squared = 0
    for floor in range(floors):
        for direction in range(1, 3):
            waiting_times_squared += sum([wait_time**2 for wait_time in building[direction, floor]])

    # Update the building state and reward system based on action
    if action == 0:  # Move up
        if directionFloorOccupants[1] < floors - 1:
            directionFloorOccupants[1] += 1
        else:
            reward -= 1  # Penalty for invalid action

    elif action == 1:  # Move down
        if directionFloorOccupants[1] > 0:
            directionFloorOccupants[1] -= 1
        else:
            reward -= 1  # Penalty for invalid action

    elif action == 2:  # Stop
        reward -= waiting_times_squared  # Minimize waiting times squared
        building[1:, directionFloorOccupants[1]] = 0  # Clear served requests

    # Check if simulation ends (e.g., no requests left)
    done = np.sum(building[1:]) == 0

    # Create next state
    next_state = np.concatenate((building.flatten(), directionFloorOccupants)).reshape(1, -1)
    return next_state, reward, done

if __name__ == "__main__":
    main()
