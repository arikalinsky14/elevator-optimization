#RANDOM NOt_E SO IMA PUT IT HERE: BOTH ALGOS ALLOW AN ELEVATOR TO DROP OFF, SWITCH DIRECTION AND PICK UP IN SAME TIME STEP. THIS DONE BC
#DIRECTION DOES NOT HAVE TO EQUAL PASSANGER REQUEST DIRECTION (for max it must be 0, for greedy it can be -1 and 1)


import random
import numpy as np
import math


#IN THE FUTURE, maybe make it calculatate average waiitng time per floor direction. Just need to keep track of how many users generated per slot, and add number for each timestep

def main():
    #INDEPENDENT VARIABLES; CAN ALSO EDIT PERSON GENERATION
    floors = 10
    carCapacity = 6
    totalTime = 10000

    #Metric tracking vars
    pplGenerated = 0
    totalPeopleWaitingTicks = 0
    numWaitingArray = [0] * totalTime
    #For wait time squared, create empty list for floor and direction for waittimes
    hallCalls = np.empty([3, floors], dtype=object)
    for i in range(0,3):
        for j in range(floors):
            hallCalls[i][j] = []
    waitingTimesSquared = [0]

    #Simulation vars
    carRequests = [-1] * carCapacity
    directionFloorOccupants = [0,0,0]
    #Above list has elevator's direction, floor, and occupant number in said order.


    #building will have a matrix with elevator information
    #row 0 lists the floor number
    #row 1 is number of up floor requests
    #row 2 is number of down floor requests
    building = np.zeros([3,floors])
    for x in range(0,floors):
        building[0][x] = x
    
    
    outfile = open("greedyElevator.txt",'w')

    for time in range(0,totalTime):
        print("")
        outfile.write(f"\n\n\nTime step: {time + 1}")
        outfile.write(f"\n")
        print("Time step:", time+1)

        #Add timestep
        for f in range(0,floors):
            for l in range(1,3):
                numWaitingArray[time] = building[l][f] + numWaitingArray[time]
                for x in range(len(hallCalls[l][f])): #For every time step, add 1 time step waited for each hall call 
                    hallCalls[l][f][x] += 1
        totalPeopleWaitingTicks = totalPeopleWaitingTicks + numWaitingArray[time]         

        #randomly generate people
        pplGenerated = peopleGeneration(pplGenerated, building, hallCalls)

       
        continueRun(building,carRequests,directionFloorOccupants,floors, carCapacity, hallCalls, waitingTimesSquared)
        if(directionFloorOccupants[0] == 0 and directionFloorOccupants[2] == 0):
            greedyAlgorithim(building,directionFloorOccupants,floors)
            outfile.write(f"Algo called: {directionFloorOccupants[0]}\n")
            #print("Algorithim called")


        for r in range(0,3):
            for c in range(0,floors):
                #print(building[r][c], end=" ")
                outfile.write(f"{building[r][c]} ")
            if(r == 0):
                #print("  Floor")
                outfile.write(f"   Floor\n")
            if(r == 1):
                #print("  Up Requests per Floor")
                outfile.write(f"   Up Requests per Floor \n")
            if(r == 2):
                #print("  Down Requests per Floor")
                outfile.write(f"   Down Requests per Floor\n")
        
        for r in range(0,3):
            for c in range(0,floors):
                #print(building[r][c], end=" ")
                outfile.write(f"{len(hallCalls[r][c])} ")
            if(r == 0):
                #print("  Floor")
                outfile.write(f"   Floor\n")
            if(r == 1):
                #print("  Up Requests per Floor")
                outfile.write(f"   Up Requests per Floor \n")
            if(r == 2):
                #print("  Down Requests per Floor")
                outfile.write(f"   Down Requests per Floor\n")
        #print("Elevator direction: ", directionFloorOccupants[0])
        outfile.write(f"Elevator direction: {directionFloorOccupants[0]} \n")
        #print("Elevator floor: ", directionFloorOccupants[1])
        outfile.write(f"Elevator floor: {directionFloorOccupants[1]} \n")
        #print("Elevator occupants: ", directionFloorOccupants[2])
        outfile.write(f"Elevator occupants: {directionFloorOccupants[2]} \n")
        #print("Cabin calls:", end=" ")
        outfile.write(f"Cabin calls: ")
        for n in range(0,carCapacity):
            #print(carRequests[n], end= " ")
            outfile.write(f"{carRequests[n]} ")
        #print(" ")
        #print(" ")

    #Time simulation is over, add waiting times of people still in building
    for l in range(1,3):
        for f in range(floors):
            for x in range(len(hallCalls[l][f])):
                hallCalls[l][f][x] += 1
    
    #print("Final Stats: ")
    outfile.write(f"\n\nFinal Stats: \n")
    #print("Average number of people waiting: ", totalPeopleWaitingTicks/totalTime)
    outfile.write(f"Average number of people waiting: {totalPeopleWaitingTicks/totalTime}\n")
    #print("People generated:", pplGenerated)
    outfile.write(f"People generated: {pplGenerated}\n")
    #print("Average waiitng time per person:", totalPeopleWaitingTicks/pplGenerated)
    outfile.write(f"Average number of people waiting: {totalPeopleWaitingTicks/pplGenerated}\n")
    outfile.write(f"Avg waiting time squared: {math.sqrt(waitingTimesSquared[0]/pplGenerated)}")

    outfile.close()


#Function is the decision maker of the elevator algorithim. Runs if elevator is idle
def greedyAlgorithim(building,directionFloorOccupants,floors):
    closestRequest = floors*3
    downOrUp = 0

    #Loop finds which floor with a request is closest
    for r in range (1,3):
        for c in range(0, floors):
            if(building[r][c] != 0):
                if(abs(c - directionFloorOccupants[1]) <= abs(directionFloorOccupants[1] - closestRequest)):
                    closestRequest = c
                    downOrUp = r
    #Return direction of closestRequest or 0 if no requests (or same floor requests)
    if(closestRequest == floors*3):
        directionFloorOccupants[0] = 0
    elif(closestRequest > directionFloorOccupants[1]):
        directionFloorOccupants[0] = 1
    elif(closestRequest < directionFloorOccupants[1]):
        directionFloorOccupants[0] = -1
    elif(closestRequest == directionFloorOccupants[1]):
        if(downOrUp == 1):
            directionFloorOccupants[0] = 1
        else:
            directionFloorOccupants[0] = -1

        
                

def continueRun(building, carRequests, directionFloorOccupants,floors,carCapcity, hallCalls, waitingTimesSquared):
    elevatorDirection = directionFloorOccupants[0]
    elevatorFloor = directionFloorOccupants[1]
    elevatorOccupants = directionFloorOccupants[2]
    Stopped = False

    #Check if anyone on elevator would like to leave
    for c in range(0,carCapcity):
        if(carRequests[c] == elevatorFloor):
            carRequests[c] = -1
            elevatorOccupants = elevatorOccupants-1
            #if someone leaves and no occupants, direction is idle
            if(elevatorOccupants == 0):
                elevatorDirection = 0
            Stopped = True
    
    #Remove any middle -1's
    c = 0
    while(c < carCapcity):
        if(carRequests[c] != -1):
            for i in range(0,carCapcity-1):
                if(carRequests[i] == -1 and i < c):
                    carRequests[i] = carRequests[c]
                    carRequests[c] = -1
                    
        c = c + 1
    
    #Check if anyone on current floor would like to enter (if elevator going up)
    while(elevatorOccupants < carCapcity and building[1][elevatorFloor] > 0 and elevatorDirection == 1):
        carRequests[elevatorOccupants] = random.randint(elevatorFloor+1,floors-1)
        elevatorOccupants = elevatorOccupants + 1
        building[1][elevatorFloor] = building[1][elevatorFloor] - 1
        Stopped = True
        waitingTimesSquared[0] += (hallCalls[1][elevatorFloor].pop(0)) ** 2


    #Check if anyone on current floor would like to enter (if elevator going down)
    while(elevatorOccupants < carCapcity and building[2][elevatorFloor] > 0 and elevatorDirection == -1):
        carRequests[elevatorOccupants] = random.randint(0,elevatorFloor-1)
        elevatorOccupants = elevatorOccupants + 1
        building[2][elevatorFloor] = building[2][elevatorFloor] - 1
        Stopped = True
        waitingTimesSquared[0] += (hallCalls[2][elevatorFloor].pop(0)) ** 2

    #If elevator is empty it changes direction to service said request (PART OF GREEDY ALGO)
    if(elevatorOccupants == 0):
        while(elevatorOccupants < carCapcity and building[1][elevatorFloor] > 0):
            elevatorDirection = 1
            carRequests[elevatorOccupants] = random.randint(elevatorFloor+1,floors-1)
            elevatorOccupants = elevatorOccupants + 1
            building[1][elevatorFloor] = building[1][elevatorFloor] - 1
            Stopped = True
            waitingTimesSquared[0] += (hallCalls[1][elevatorFloor].pop(0)) ** 2
    if(elevatorOccupants == 0):
        while(elevatorOccupants < carCapcity and building[2][elevatorFloor] > 0):
            elevatorDirection = -1
            carRequests[elevatorOccupants] = random.randint(0,elevatorFloor-1)
            elevatorOccupants = elevatorOccupants + 1
            building[2][elevatorFloor] = building[2][elevatorFloor] - 1
            waitingTimesSquared[0] += (hallCalls[2][elevatorFloor].pop(0)) ** 2
            Stopped = True

    #Stop elevator if at min or max floor
    if(elevatorFloor == floors-1 and elevatorDirection == 1):
        elevatorDirection = 0
        Stopped = True
    if(elevatorFloor == 0 and elevatorDirection == -1):
        elevatorDirection = 0
        Stopped = True

    #Elevator moves if its not already at a max/min
    if(not Stopped):
        elevatorFloor = elevatorFloor + elevatorDirection

    directionFloorOccupants[0] = elevatorDirection
    directionFloorOccupants[1] = elevatorFloor
    directionFloorOccupants[2] = elevatorOccupants

def peopleGeneration(pplGenerated, building, hallCalls):
    floors = len(building[0])
    for f in range(0,floors):
        for l in range(1,3):
            generation = np.random.poisson(0.025)
            if(generation + building[l][f] >= 4):
                generation = 4 - building[l][f]
            pplGenerated += generation
            building[l][f] = building[l][f] + generation
            while generation > 0: #append a new "person" index for each generated person
                hallCalls[l][f].append(0)
                generation = generation - 1
    building[1][floors-1] = 0
    building[2][0] = 0
    hallCalls[1][floors - 1] = []
    hallCalls[2][0] = []
    return pplGenerated


if(__name__ == "__main__"):
    main()