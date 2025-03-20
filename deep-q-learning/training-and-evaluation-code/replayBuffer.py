from collections import deque
from collections import namedtuple
import random
import tensorflow as tf
import numpy as np

# Replay buffer class; used to save a memory of past episodes so that we can do mini-batch training
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state):
        e = self.experience(state, action, reward, next_state) #All past experiences are stored in a deque, where each element is a namedTuple of the state transition
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size) #Array of (batch_size) experiences, each element is stll a namedTuple

        states = tf.convert_to_tensor(np.vstack([e.state for e in experiences if e is not None]), dtype=tf.float64) #A tensor is made from the array of namedTuplies we train with
        actions = tf.convert_to_tensor(np.vstack([e.action for e in experiences if e is not None]), dtype=tf.float64) #Rows correspond to state_1, state_2, etc.
        rewards = tf.convert_to_tensor(np.vstack([e.reward for e in experiences if e is not None]), dtype=tf.float64) #Columns donote seperate experience
        next_states = tf.convert_to_tensor(np.vstack([e.next_state for e in experiences if e is not None]), dtype=tf.float64) #This is the case for all 4 tensors made

        return (states, actions, rewards, next_states) #Return an UNNAMED regular tuple of the batch (in format described above)

    def __len__(self):
        return len(self.memory)