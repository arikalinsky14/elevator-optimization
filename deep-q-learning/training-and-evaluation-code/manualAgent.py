from elevatorEnvironment import Elevator
from replayBuffer import ReplayBuffer

def main():
    thackery = Elevator(floors=10,capacity=5)
    for time in range(0,100):
        print(f"\n\n---------Time: {time}---------")
        print(thackery.building)
        print("Building:")
        
        x = [1,2,3,4,5,6,7,8,9,10]
        print("Floor:  " + " ".join(map(str, x))) 

        print("Up:     " + " ".join(map(str, thackery.buildingTotals[0]))) 
        print("Down:   " + " ".join(map(str, thackery.buildingTotals[1]))) 
            
        print("\nElevator floor: ", thackery.elevatorFloor)
        print("Last policy: ", thackery.lastPolicy)
        print("Car Calls: " + "   ".join(map(str, thackery.carCalls)))
        #print("Total Loss Function: ", thackery.computeTotalValue())
        print("Timestep Loss Function:", thackery.computeValue())
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



if __name__ == "__main__":
    main()