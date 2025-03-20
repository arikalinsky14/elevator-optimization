from elevatorEnvironment import Elevator
import tensorflow as tf
import os
import numpy as np
import time
import random
import pandas as pd

def main():
    df_loc = r"C:\Users\arika\OneDrive\Desktop\Research Papers\evalTest.csv"
    df_loc2 = r"C:\Users\arika\OneDrive\Desktop\Research Papers\numReqs.csv"
    currentResults = pd.read_csv(df_loc)
    trial = "neural-6"

    results = np.zeros(1000) 
    numReqs = np.zeros(1000)

    path = r"C:\Users\arika\OneDrive\Desktop\Python Code\savedModels\fiveCapacityMoreTraining"
    
    agentDQN = tf.keras.models.load_model(path)

    for x in range(0, 1000):
        thackery = Elevator(floors=10, 
                            capacity=5, 
                            maximumUsers=100, 
                            poissonLambda= 0.01)

        for timeStep in range(0,1000):
            thackery.newArrivals()
            if(trial == "neural-6"):
                state = tf.convert_to_tensor(thackery.returnState(), tf.float64)
                
                # If state is a single example, add a batch dimension.
                if len(state.shape) == 1:
                    state = tf.expand_dims(state, axis=0)  # Now shape is (1, 108)
                Q_scores = agentDQN(state, training=False)
                action = np.argmax(tf.squeeze(Q_scores)) - 1
                #action = thackery.greedyControl()
            elif(trial == "random-neural-.1"):
                state = tf.convert_to_tensor(thackery.returnState(), tf.float64)
                
                # If state is a single example, add a batch dimension.
                if len(state.shape) == 1:
                    state = tf.expand_dims(state, axis=0)  # Now shape is (1, 108)
                Q_scores = agentDQN(state, training=False)
                action = np.argmax(tf.squeeze(Q_scores)) - 1
                if (random.random() < .1):
                    action = random.choice([-1, 0, 1])
            elif(trial == "random-neural-.025"):
                state = tf.convert_to_tensor(thackery.returnState(), tf.float64)
                
                # If state is a single example, add a batch dimension.
                if len(state.shape) == 1:
                    state = tf.expand_dims(state, axis=0)  # Now shape is (1, 108)
                Q_scores = agentDQN(state, training=False)
                action = np.argmax(tf.squeeze(Q_scores)) - 1
                if (random.random() < .025):
                    action = random.choice([-1, 0, 1])
            elif(trial == "random-neural-.01"):
                state = tf.convert_to_tensor(thackery.returnState(), tf.float64)
                
                # If state is a single example, add a batch dimension.
                if len(state.shape) == 1:
                    state = tf.expand_dims(state, axis=0)  # Now shape is (1, 108)
                Q_scores = agentDQN(state, training=False)
                action = np.argmax(tf.squeeze(Q_scores)) - 1
                if (random.random() < .01):
                    action = random.choice([-1, 0, 1])
            elif(trial == "random-neural-.05"):
                state = tf.convert_to_tensor(thackery.returnState(), tf.float64)
                
                # If state is a single example, add a batch dimension.
                if len(state.shape) == 1:
                    state = tf.expand_dims(state, axis=0)  # Now shape is (1, 108)
                Q_scores = agentDQN(state, training=False)
                action = np.argmax(tf.squeeze(Q_scores)) - 1
                if (random.random() < .05):
                    action = random.choice([-1, 0, 1])
            elif(trial == "maximum"):
                action = thackery.deterministicControl()
            elif(trial == "greedy"):
                action = thackery.greedyControl()
            else:
                action = 0

            
            thackery.timeStep(action)

            totalValue = thackery.computeTotalValue()
            results[timeStep] += totalValue

            requestsHere = thackery.getCurrentUsers()
            numReqs[timeStep] += requestsHere


            if(timeStep % 750 == 0 and timeStep != 0):
                print(f"Time step: {timeStep}")
                print(f"Total Value: {totalValue}") 
                print(f"Number requests: {requestsHere}")
                print(thackery.building)
        print(f"Repetition: {x}")   

    currentResults = pd.read_csv(df_loc)
    df = pd.DataFrame(results, columns=[trial])
    df[trial] = df[trial]/1000
    df_combined = pd.concat([currentResults, df], axis=1)
    df_combined.to_csv(df_loc,index=False)

    currentResults = pd.read_csv(df_loc2)
    df = pd.DataFrame(numReqs, columns=[trial])
    df[trial] = df[trial]/1000
    df_combined = pd.concat([currentResults, df], axis=1)
    df_combined.to_csv(df_loc2,index=False)



    
    """
    for layer in agentDQN.layers:
        print(f"\nLayer: {layer.name}")
        weights = layer.get_weights()
        if weights:  # Only print layers that have weights
            for i, weight_array in enumerate(weights):
                print(f"  Weight {i}:")
                print(weight_array)
        else:
            print("  (No weights)")
    """


    





if __name__ == "__main__":
    main()