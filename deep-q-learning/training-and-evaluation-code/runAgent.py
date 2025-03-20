from elevatorEnvironment import Elevator
import tensorflow as tf
import os
import numpy as np
import time

def main():
    name = "2-27"
    thackery = Elevator(floors=10, 
                        capacity=5, 
                        maximumUsers=10, 
                        poissonLambda= 0.025)
    
    path = os.path.join("..", "savedModels", name)
    path = r"C:\Users\arika\OneDrive\Desktop\Python Code\savedModels\fiveCapacityMoreTraining"
    agentDQN = tf.keras.models.load_model(path)

    for timeStep in range(0,10000):
        thackery.newArrivals()

        print(f"\n\n---------State Transition: {timeStep}---------")
        print(f"Current Users: {thackery.currentUsers}")
        
        first_state = thackery.returnState()
        print(f"Len: {len(first_state)}")
        #print(first_state)
        #print(f"Length of tensor: {len(first_state)}")

        print("Building:")
        
        x = range(1, 11)
        print("Floor:  " + " ".join(map(str, x))) 

        print("Up:     " + " ".join(map(str, thackery.buildingTotals[0]))) 
        print("Down:   " + " ".join(map(str, thackery.buildingTotals[1]))) 
            
        print("\nElevator floor: ", thackery.elevatorFloor)
        print("Car Calls: " + "   ".join(map(str, thackery.carCalls)))
        print()

        #Run network for input
        # Ensure all tensors are float64 (double)
        state = tf.convert_to_tensor(thackery.returnState(), tf.float64)
        
        # If state is a single example, add a batch dimension.
        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)  # Now shape is (1, 108)
        Q_scores = agentDQN(state, training=False)
        print(f"Q-scores: {tf.squeeze(Q_scores)}")
        action = np.argmax(tf.squeeze(Q_scores)) - 1
        
        print(f"Policy: {action}")
        thackery.timeStep(action)

        print(f"State-Action Initial Reward: {thackery.computeValue()}\n")    

        second_state = thackery.returnState()
        #print(first_state)
        #print(f"Length of tensor: {len(first_state)}")

        print("Building:")
        
        x = [1,2,3,4,5,6,7,8,9,10]
        print("Floor:  " + " ".join(map(str, x))) 

        print("Up:     " + " ".join(map(str, thackery.buildingTotals[0]))) 
        print("Down:   " + " ".join(map(str, thackery.buildingTotals[1]))) 
            
        print("\nElevator floor: ", thackery.elevatorFloor)
        print("Car Calls: " + "   ".join(map(str, thackery.carCalls)))
        print()
        time.sleep(.5)

    
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