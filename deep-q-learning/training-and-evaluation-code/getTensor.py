from elevatorEnvironment import Elevator
import tensorflow as tf

def main():
    thackery = Elevator(floors=10,capacity=1,maximumUsers=2,poissonLambda=.025)
    for time in range(0,100):
        thackery.newArrivals()

        print(f"\n\n---------State Transition: {time}---------")
        print(f"Current Users: {thackery.currentUsers}")
        
        first_state = thackery.returnState()
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

        # Prompt the user to input an integer

        user_input = input("Please enter a policy (-1,0,1): ")
        

        # Convert the input to an integer
        try:
            policy = int(user_input)
            print(f"You entered: {policy}")
            thackery.timeStep(policy)
        except ValueError:
            print("That's not a valid integer.")
            thackery.timeStep(0)


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




if __name__ == "__main__":
    main()