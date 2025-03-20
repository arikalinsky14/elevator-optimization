from elevatorEnvironment import Elevator
from tfAgent import ControlAgent
import tensorflow as tf
import numpy as np
import random


def main():
    print("TRAINING LOOP STARTED")
    #File names
    save_file = "FiveCapacityNew"
    save_count = 0
    training_log = r"C:\Users\arika\OneDrive\Desktop\Python Code\trainingReports\log-" + save_file

    #Define training hyperparemeters
    trainingSteps: int = 500000
    epsilon = 1 #epsilon max is 1
    epsilon_end = .1
    epsilon_decay = .995
    updateFrequency = 500 ##How often LR is updated
    targetUpdatePeriod = 1000 #How often the target DQN is updated (And how often results are printed)

    #Create an elevator object, this is the simulation of the building. Input building hyperparameters
    thackery = Elevator(floors=10,capacity=5,maximumUsers=1,poissonLambda=.05)

    #Create an agent object, this class learns to control the elevator
    state_size = len(thackery) 
    action_size = 3 
    lr = 0.0025 #.0025 max
    minlr = 0.0003
    gamma = .99 
    seed = 37
    batch_size = 128

    agent = ControlAgent(state_size, action_size, lr, minlr, updateFrequency, gamma, seed, batch_size, targetUpdatePeriod, training_log)
    #agent.newNetwork()
    try:
        agent.loadNetwork(save_file)
    except Exception as e:
        print(f"Network cannot be read: {e}")
        agent.newNetwork()

    for time in range(0,trainingSteps + 1000):
        thackery.newArrivals()
        first_state = thackery.returnState()
        if(random.random() < epsilon):
            action = random.choice([-1, 0, 1])
        else:
            action = np.argmax(tf.squeeze(agent.getQValues(state=first_state))) - 1
            #if(time >= 100000):
            #    print(f"New timestep generation: Chose {action} from {tf.squeeze(agent.getQValues(state=first_state))}")
        thackery.timeStep(action)
        qScore = thackery.computeValue()
        second_state = thackery.returnState()
        agent.attemptLearn(state=first_state,action=action,reward=qScore,next_state=second_state,epsilon=epsilon)

        if(time % updateFrequency == 0 and time >= 1000):
            epsilon = max(epsilon*epsilon_decay,epsilon_end)
            print(f"Epsilon: {epsilon}")
            if(time % (targetUpdatePeriod * 5) == 0):
                agent.saveNetwork(save_file)
                save_count += 1
                print(f"Save number: {save_count}")
                #agent.print_network_weights()
                #print(f"Max Requests: {maximumUseers}")
        
    print("All losses: ")
    print(agent.returnAllLoses())

    agent.saveNetwork(save_file)
    print(f"Save number: {save_count}")
        


main()