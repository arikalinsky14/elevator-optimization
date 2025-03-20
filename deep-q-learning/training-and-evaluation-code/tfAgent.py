import tensorflow as tf
import numpy as np
from collections import deque
from collections import namedtuple
from replayBuffer import ReplayBuffer
import random
import os


class ControlAgent:
    def __init__(self, 
                 state_size, 
                 action_size, 
                 lr, 
                 minlr, 
                 lrlife, 
                 gamma, 
                 seed, 
                 batch_size, 
                 targetUpdatePeriod = 1000, 
                 filePath = r"C:\Users\arika\OneDrive\Desktop\Python Code\trainingReports\lastestLog"):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.lr = lr
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.lr)
        self.minlr = minlr
        self.lrlife = lrlife
        self.targetUpdatePeriod = targetUpdatePeriod

        self.gamma = gamma
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.bufferSize = int(40000)
        self.memory = ReplayBuffer(action_size,buffer_size=self.bufferSize, batch_size = batch_size, seed=seed)
        self.actualLearns = 0

        self.localDQN = None
        self.targetDQN = None

        self.recentLoss = 0
        self.allLosses = []
        self.upCounter: int = 0
        self.downCounter: int = 0
        self.stayCounter: int = 0

        self.filename = filePath

    def returnAllLoses(self):
        return self.allLosses

    def loadNetwork(self,name):
        path = os.path.join(r"C:\Users\arika\OneDrive\Desktop\Python Code\savedModels", name)
        self.localDQN = tf.keras.models.load_model(path)
        self.targetDQN = tf.keras.models.load_model(path)
        # Recompile the model
        self.localDQN.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['mse'])
        self.targetDQN.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['mse'])
        print("NETWORKS LOADED")

    def saveNetwork(self,name):
        path = os.path.join("..", "savedModels", name)
        self.localDQN.save(path)
        print(f"Local DQN model saved to {path}")

    def newNetwork(self):
        self.localDQN = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_size,), dtype=tf.float64),
            tf.keras.layers.Dense(64, activation="elu", dtype=tf.float64),
            tf.keras.layers.Dropout(0.2),  # Prevent overfitting
            tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            tf.keras.layers.Dense(self.action_size, dtype=tf.float64)
        ])
        self.targetDQN = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_size,), dtype=tf.float64),
            tf.keras.layers.Dense(64, activation="elu", dtype=tf.float64),
            tf.keras.layers.Dropout(0.2),  # Prevent overfitting
            tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            #tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
            tf.keras.layers.Dense(self.action_size, dtype=tf.float64)
        ])
        
        self.localDQN.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['mse'])
        self.targetDQN.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['mse'])
        
    def print_learning_states(self,epsilon):
        totalPolicies = self.upCounter + self.downCounter + self.stayCounter
        with open(self.filename, 'a') as file:  # Use 'a' to append instead of overwrite
            file.write(f"Learning step: {self.actualLearns}\n")
            file.write(f"Epsilon: {epsilon}\n")
            file.write(f"LR: {self.lr}\n")
            file.write(f"Recent loss: {self.recentLoss/self.targetUpdatePeriod}\n")
            file.write(f"Policy percents in latest training: \n")
            file.write(f"Up: {self.upCounter/totalPolicies} Stay: {self.stayCounter/totalPolicies} Down: {self.downCounter/totalPolicies} \n\n")


        print(f"Learning step: {self.actualLearns}")
        print(f"Recent loss: {self.recentLoss/self.targetUpdatePeriod}")
        print(f"LR: {self.lr}")

        self.allLosses.append([self.recentLoss])
        self.recentLoss = 0

        self.upCounter = 0
        self.downCounter = 0
        self.stayCounter = 0
        

        


    def print_network_weights(self):
        """
        Prints the weights for each layer in the given Keras model.

        Parameters:
        - model: A tf.keras.Model object.
        """
        for layer in self.localDQN.layers:
            print(f"\nLayer: {layer.name}")
            weights = layer.get_weights()
            if weights:  # Only print layers that have weights
                for i, weight_array in enumerate(weights):
                    print(f"  Weight {i}:")
                    print(weight_array)
            else:
                print("  (No weights)")


    #Basically the main of thie class
    #Attempt to learn; a new state is required so agent gets breadth of experience
    #Call this function and the agent tries to do a training batch; if it is able to train it calls another function
    #This also updates hyperparameters
    def attemptLearn(self, state,action,reward,next_state,epsilon): 
        self.memory.add(state,action,reward,next_state) #add the new input to memory
        if(len(self.memory) >= 1000): #Train if resonable memory is stored already
            self.actualLearns += 1
            experiences = self.memory.sample() #Fetches an unnamed table of (states, actions, rewards, next_states); each is a 2d matrix; rows are unique experiences
            self.recentLoss += self.learn(experiences)
            #Every lrLife time steps, lr is decreased by 1%, or kept at minimum
            if(self.actualLearns % self.lrlife == 0):
                self.lr = max(self.lr * .99, self.minlr)
                self.optimizer.learning_rate.assign(self.lr)
            if(self.actualLearns % (self.targetUpdatePeriod) == 0): #we update the target DQN at set intervals and display stats
                self.print_learning_states(epsilon) #Print learning progress to a file
                self.targetDQN.set_weights(self.localDQN.get_weights())
                print(f"Local DQN is now target DQN")
                


    def getQValues(self,state):
        # Ensure all tensors are float64 (double)
        #states = tf.convert_to_tensor(state, tf.float64)
        
        # If state is a single example, add a batch dimension.
        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)  # Now shape is (1, len(state))
        Q_scores = self.localDQN(state, training=False)
        return Q_scores


    def learn(self, experiences):
        if(self.actualLearns % 1000  == 0):
            print(f"Learning time: {self.actualLearns}")
        states, actions, rewards, next_states = experiences #Unzips experience tuple to individual arrays; rows correspond to experience number 

        # Ensure all tensors are float64 (double)
        states = tf.convert_to_tensor(states, tf.float64)
        rewards = tf.convert_to_tensor(rewards, tf.float64)
        next_states = tf.convert_to_tensor(next_states, tf.float64)
        #print("Rewards vector: ")
        #print(rewards)
        #print("state 1")
        #print(states[0])
        #print("state 2")
        #print(next_states[0])

        #print(type(actions))

        #count number of each action we train on
        self.downCounter += tf.reduce_sum(tf.cast(tf.equal(actions, -1), tf.int32))
        self.stayCounter += tf.reduce_sum(tf.cast(tf.equal(actions, 0), tf.int32))
        self.upCounter += tf.reduce_sum(tf.cast(tf.equal(actions, 1), tf.int32))

        # Convert actions to correct range ([-1,0,1] -> [0,1,2])
        actions = actions + 1

        # (Double DQN):
        best_actions = tf.argmax(self.localDQN(next_states), axis=1)  # Action selection via LOCAL network
        Q_target_next_all = self.targetDQN(next_states, training=False)  # Q-value evaluation via TARGET network
        Q_target_next = tf.gather(Q_target_next_all, best_actions, batch_dims=1, axis=1)
        Q_target_next = tf.cast(Q_target_next, tf.float64)

        #print("Q_target_next_all")
        #print(Q_target_next_all)
        #print("Q_target_next")
        #print(Q_target_next)

        # Compute the target Q-values using the Bellman equation:
        Q_targets = rewards + self.gamma * Q_target_next #tensorflow auto does element-wise operations to all samples in column vector
        Q_targets = tf.cast(Q_targets, tf.float64)

        #Use GradientTape for automatic differentiation. This records the operations done to the reward function so it knows how to compute the gradient.
        #Target network is not updated in GradientTime since for the time being we consider it a constant. Q_targets isnt based on trainanble parameters
        with tf.GradientTape() as tape:
            # Forward pass on the local network to compute Q-values for the current states.
            # This outputs a tensor of shape [batch_size, action_size] (action_size is 3 for your 3 actions).
            Q_expected_all = tf.cast(self.localDQN(states, training=False), tf.float64)

            # We now need to extract the Q-value corresponding to the action that was taken.
            # One common way is to convert the actions to one-hot vectors and then multiply element-wise.
            actions_onehot = tf.one_hot(tf.squeeze(tf.cast(actions, tf.int32), axis=1), depth=self.action_size, dtype=tf.float64) #We squeeze actions to a row vector since one_hot wants this input, we also force it to be int
            # Multiply each predicted Q-value by the one-hot action vector and sum over the action dimension.
            # This effectively picks the Q-value for the action taken.
            Q_expected = tf.reduce_sum(Q_expected_all * actions_onehot, axis=1, keepdims=True)

            # Compute the loss between the target Q-values (from the Bellman equation) and the predicted Q-values.
            # We use Mean Squared Error (MSE) as the loss.   1/N sigma(Q_target_i - t_expected_i)^2
            loss = self.loss_fn(Q_targets, Q_expected)
            #print(f"Loss function: {loss}")
        
        # Compute gradients of the loss with respect to the trainable parameters of the local network.
        gradients = tape.gradient(loss, self.localDQN.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 3.0)

        # Apply the gradients to update the local network's parameters. #Gradients is a tensor that has as many indexes as model weights, zip paits the gradient with the trainable variables. Two 1d lists are merged into a tuple
        self.optimizer.apply_gradients(zip(gradients, self.localDQN.trainable_variables))

        # Optionally return the loss as a NumPy scalar for logging or debugging purposes.
        return loss.numpy()
    


