from tfAgent import ElevatorControl
import tensorflow as tf

def main():
    #EGC = ElevatorControl(state_size=108,action_size=3,seed=37)
    #learn()
    states = tf.constant([[1,2,3],[4,5,6]])
    actions = tf.constant([[1],[2]])
    rewards = tf.constant([[1],[2]])
    next_states = tf.constant([[1,2,3],[4,5,6]])

    experiences = (states, actions, rewards, next_states)



    state, actions, rewards, next_states = experiences #Unzips experience tuple to individual arrays; rows correspond to experience number (* is unpacking operator)
    print(type(state))

main()