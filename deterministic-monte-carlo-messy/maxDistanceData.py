import random
import numpy as np
import math

def main():
    #OUTPUT MATRICES
    avgPplWaiting = np.zeros((10,8))
    avgWaitingTime = np.zeros((10,8))
    avgWaitingTimeSquared = np.zeros((10,8))

    for floors in range(3,12):
        for carCapacity in range(3,9):
            print("Floor:", floors)
            print("Car Capacity:", carCapacity)
            print(" ")

            #Independent variables
            totalTime = 1000000

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

            for time in range(0,totalTime):
                for f in range(0,floors):
                    for l in range(1,3):
                        numWaitingArray[time] = building[l][f] + numWaitingArray[time]
                        for x in range(len(hallCalls[l][f])):
                            hallCalls[l][f][x] += 1
                totalPeopleWaitingTicks = totalPeopleWaitingTicks + numWaitingArray[time]

                #randomly generate people
                pplGenerated = peopleGeneration(pplGenerated, building, hallCalls)
            
                continueRun(building,carRequests,directionFloorOccupants,floors, carCapacity, hallCalls, waitingTimesSquared)
                if(directionFloorOccupants[0] == 0):
                    generousAlgorithim(building,directionFloorOccupants,floors)
            
            #Time simulation is over, add waiting times of people still in building
            for l in range(1,3):
                for f in range(floors):
                    for x in range(len(hallCalls[l][f])):
                        hallCalls[l][f][x] += 1

            
            avgPplWaiting[floors-2][carCapacity-1] = totalPeopleWaitingTicks/totalTime
            avgWaitingTime[floors-2][carCapacity-1] = totalPeopleWaitingTicks/pplGenerated
            avgWaitingTimeSquared[floors-2][carCapacity-1] = math.sqrt(waitingTimesSquared[0]/pplGenerated)
    print("Avg PPL Waiting:")
    for r in range(0,10):
        for c in range(0,8):
            print(avgPplWaiting[r][c], end=" ")
        print(" ")
    print(" ")
    print("Avg Waiting Time")
    for r in range(0,10):
        for c in range(0,8):
            print(avgWaitingTime[r][c], end=" ")
        print(" ")
    print("Sqrt of Avg Waiting Time squared")
    for r in range(0,10):
        for c in range(0,8):
            print(avgWaitingTimeSquared[r][c], end=" ")
        print(" ")


#Function is the decision maker of the elevator algorithim. Runs if elevator is idle
def generousAlgorithim(building,directionFloorOccupants,floors):
    furthestRequest = directionFloorOccupants[1]

    #Loop finds which floor with a request is furthest
    for r in range (1,3):
        for c in range(0, floors):
            if(building[r][c] != 0):
                if(abs(c - directionFloorOccupants[1]) >= abs(c - furthestRequest)):
                    furthestRequest = c
    #Return direction of furthestRequest or 0 if no requests (or same floor requests)
    if(furthestRequest > directionFloorOccupants[1]):
        directionFloorOccupants[0] = 1
    elif(furthestRequest < directionFloorOccupants[1]):
        directionFloorOccupants[0] = -1
    else:
        #directionFloorOccupants[0] = 0
        if(building[1][furthestRequest] > 0):
            directionFloorOccupants[0] = 1
        elif(building[2][furthestRequest] > 0):
            directionFloorOccupants[0] = -1
        else:
            directionFloorOccupants[0] = 0
                

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
    
    #check if elevator still has requests
    if(elevatorDirection == -1 and elevatorOccupants == 0):
        requestsBelow = False
        for i in range(0,elevatorFloor):
            if(building[1][i] > 0 or building[2][i] > 0):
                requestsBelow = True
        if(not requestsBelow):
            elevatorDirection = 0

    if(elevatorDirection == 1 and elevatorOccupants == 0):
        requestsAbove = False
        for i in range(elevatorFloor+1,floors):
            if(building[1][i] > 0 or building[2][i] > 0):
                requestsAbove = True
        if(not requestsAbove):
            elevatorDirection = 0

    #Check if anyone on current floor would like to enter (if elevator idle), must be checked now to match greedy case
    if(elevatorOccupants < carCapcity and building[1][elevatorFloor] > 0 and elevatorDirection == 0):
        elevatorDirection = 1
        carRequests[elevatorOccupants] = random.randint(elevatorFloor+1,floors-1)
        elevatorOccupants = elevatorOccupants + 1
        building[1][elevatorFloor] = building[1][elevatorFloor] - 1
        Stopped = True
        waitingTimesSquared[0] += (hallCalls[1][elevatorFloor].pop(0)) ** 2
    if(elevatorOccupants < carCapcity and building[2][elevatorFloor] > 0 and elevatorDirection == 0):
        elevatorDirection = -1
        carRequests[elevatorOccupants] = random.randint(0,elevatorFloor-1)
        elevatorOccupants = elevatorOccupants + 1
        building[2][elevatorFloor] = building[2][elevatorFloor] - 1
        Stopped = True
        waitingTimesSquared[0] += (hallCalls[2][elevatorFloor].pop(0)) ** 2
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



    #Elevator moves
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