import random
import numpy as np
import math



class Elevator:
    def __init__(self, floors, capacity, maximumUsers = 100, poissonLambda = .025):
        #Assume car capacity is also equal to floor-direction waiting capacity
        self.floors: int = floors
        self.capacity: int = capacity
        self.maximumUsers: int = maximumUsers
        self.currentUsers: int = 0
        self.poissonLambda = poissonLambda

        #buildings array will track wait time of each call, buildingTotals tracks people waiting on each floor
        self.building = np.zeros((2,self.floors,self.capacity), dtype=int)
        self.buildingTotals = np.zeros((2,self.floors), dtype=int)

        self.peopleGenerated = 0

        #Define elevator variables
        self.elevatorFloor = int(1)
        self.carCalls = np.zeros(capacity, dtype=int)
        self.lastPolicy = int(0)
        self.lastLastPolicy = int(0)

        self.lastMoveValid = True
        self.pplServedThisStep = 0

        #waitTimeSqrt
        self.allwaitTimeSquare = 0
        self.waitTimeSqrt = np.float64(0)
        self.desiredDirection = 0

        self.previousDirectionalAward: bool = False


    def timeStep(self, policy):
        self.lastMoveValid = True
        self.pplServedThisStep = 0
        self.waitTimeSqrt = 0
        if(policy == 1): #elevator goes up one floor
            if(self.elevatorFloor < self.floors): #if elevator is not already at top floor
                self.elevatorFloor += 1
            else:
                self.lastMoveValid = False
        elif(policy == -1): #policy goes down one floor
            if(self.elevatorFloor > 1): #if elevator is not already at first floor
                self.elevatorFloor += -1
            else:
                self.lastMoveValid = False
        elif(policy == 0): #elevator stays on current floor, meaning passengers can leave or enter

            ###See if anyone in elevator wants to exit to current floor
            for carRequestIndex in range(0, self.capacity):
                if(self.carCalls[carRequestIndex] == self.elevatorFloor):
                    self.carCalls[carRequestIndex] = 0
                    self.pplServedThisStep += 1
                    self.currentUsers -= 1
            self.carCalls = self.sortArray(self.carCalls)



            ###See if anyone on floor wants to enter elevator 
            if(self.lastPolicy == 1 or self.lastPolicy == 0): #if we are travelling up, allow up requests on

                for hallRequestIndex in range(0, self.capacity):
                    if(self.building[0][self.elevatorFloor-1][hallRequestIndex] > 0):

                        try:
                            #Add request to car
                            index = self.carCalls.index(0)
                            self.carCalls[index] = self.generateCarCalls(1)
                            self.pplServedThisStep += 1

                            #Add to wait time square, remove from building
                            self.waitTimeSqrt += np.float64((self.building[0][self.elevatorFloor-1][hallRequestIndex]) ** .5)
                            self.building[0][self.elevatorFloor-1][hallRequestIndex] = 0
                            self.buildingTotals[0][self.elevatorFloor-1] =  self.buildingTotals[0][self.elevatorFloor-1] - 1
                        except ValueError:
                            break
                self.building[0][self.elevatorFloor-1] = self.sortArray(self.building[0][self.elevatorFloor-1])
            if(self.lastPolicy == -1 or self.lastPolicy == 0): #if we are travelling down, allow down requests on

                for hallRequestIndex in range(0, self.capacity):
                    if(self.building[1][self.elevatorFloor-1][hallRequestIndex] > 0):

                        try:
                            #Add request to car
                            index = self.carCalls.index(0)
                            self.carCalls[index] = self.generateCarCalls(-1)
                            self.pplServedThisStep += 1

                            #Add to wait time square, remove from building
                            self.waitTimeSqrt += np.float64((self.building[1][self.elevatorFloor-1][hallRequestIndex]) ** .5)
                            self.building[1][self.elevatorFloor-1][hallRequestIndex] = 0
                            self.buildingTotals[1][self.elevatorFloor-1] += -1
                        except ValueError:
                            break
                self.building[1][self.elevatorFloor-1] = self.sortArray(self.building[1][self.elevatorFloor-1])
        self.allwaitTimeSquare += self.waitTimeSqrt ** 4
                
        

        #Up all waiting values by 1 timestep
        for l in range(0,2):
            for f in range (0,self.floors):
                for c in range(0, self.capacity):
                    if(self.building[l][f][c] > 0):
                        self.building[l][f][c] += 1
        self.lastLastPolicy = self.lastPolicy
        self.lastPolicy = policy

        
    def generateCarCalls(self, aboveOrBelow):
        if(aboveOrBelow == 1):
            return np.random.randint(self.elevatorFloor + 1,self.floors+1)
        else:
            return np.random.randint(1,self.elevatorFloor)


    def newArrivals(self):
        #Loop through each floor-direction to generate users
        for l in range(0,2):
            for f in range(0,self.floors):
                generation = np.random.poisson(self.poissonLambda) #Input is avg of number ppl generated per floor-direction 

                if(self.currentUsers + generation >= self.maximumUsers):
                    generation = self.maximumUsers - self.currentUsers
                #Reduce generation amount to be less than capacity (if needed)
                if(generation + self.buildingTotals[l][f] >= self.capacity):
                    generation = self.capacity - self.buildingTotals[l][f]

                #Add new generations to building
                self.peopleGenerated += generation
                self.buildingTotals[l][f] += generation
                self.currentUsers += generation

                # Populate the `building` matrix with `1` for each generated person, c iterates thoguh each possible slot
                c = 0
                while generation > 0 and c < self.capacity:
                    if self.building[l][f][c] == 0:  # Find the first empty slot (represented by 0)
                        self.building[l][f][c] = 1  # Mark as occupied
                        generation -= 1  # Decrease the count of people to be added
                    c += 1

            

        #Ensure there are no up requests on top floor, and no down requests on bottom floor
        if(self.buildingTotals[0][self.floors-1] > 0):
            self.currentUsers -= self.buildingTotals[0][self.floors-1]
            self.buildingTotals[0][self.floors-1] = 0
        if(self.buildingTotals[1][0] > 0):
            self.currentUsers -= self.buildingTotals[1][0]
            self.buildingTotals[1][0] = 0



        self.building[0][self.floors - 1] = np.zeros(self.capacity)
        self.building[1][0] = np.zeros(self.capacity)
    
    def sortArray(self, arr):
        """
        Sorts the given array so that all zeros are moved to the end,
        while maintaining the relative order of the nonzero elements.

        Parameters:
        arr (list): The input list of numbers.

        Returns:
        list: A new list with zeros moved to the end.
        """
        non_zero_elements = [num for num in arr if num != 0]
        zero_elements = [num for num in arr if num == 0]
        return non_zero_elements + zero_elements
    
    def computeTotalValue(self):
        newTimes = 0
        for l in range(0,2):
            for f in range (0,self.floors):
                for c in range(0, self.capacity):
                    newTimes += self.building[l][f][c] ** 2
        allTimes = self.allwaitTimeSquare + newTimes 
        if(np.isnan(math.sqrt(allTimes/self.peopleGenerated))):
            return 0
        else:
            return math.sqrt(allTimes/self.peopleGenerated)
    
    def computeValue(self):
        """
        Computes the reward for the current elevator step based on the requested floor,
        wait times, movement direction, and passengers served.
        """
        reward = 0
        timestepWaitTimes = 0

        """
        # Determine desired floor:
        #   Prioritize car requests over hall requests.
        reqFloor = 0
        row_indices, col_indices = np.where(self.buildingTotals == 1)

        # Count number of car requests.
        numCarReqs = 0
        for x in self.carCalls:
            if x != 0:
                numCarReqs += 1

        # Set desired floor based on priority.
        
        if numCarReqs >= 1:
            reqFloor = self.carCalls[0]
        elif col_indices.size > 0:
            reqFloor = col_indices[0] + 1
        else:
            reqFloor = 5
        """
        #print(f"DESIRED FLOOR: {reqFloor}")

        # Calculate detailed wait time (the result is not currently used).
        # l is the direction of request elavator button pressed
        for l in range(2):
            for f in range(self.floors):
                for c in range(self.capacity):
                    timestepWaitTimes += np.float64(self.building[l][f][c] ** 0.5)

        #print(f"FLOOR RN: {self.elevatorFloor}")
        #print(f"REQ DIR: {self.desiredDirection}")

        # Reward for staying put when appropriate.
        if self.lastPolicy == 0:
            #if self.pplServedThisStep == 0 and (np.sum(self.buildingTotals) > 0 or np.any(self.carCalls)) and reqFloor != self.elevatorFloor:
            if(self.desiredDirection != 0):
                reward -= 6  # Penalty for unproductive stay.WAS 5
                #print("Unproductive stay")
            else:
                reward += 2  # Reward for a productive stay.
                #print("Productive stay")

        # Directional reward/penalty.
        if(self.lastPolicy != 0):
            if self.lastPolicy == self.desiredDirection:
                reward += 3 #was 4 
                #print("Reward for productive move")
            elif self.lastPolicy != 0:
                reward -= 4 #was 3 in first working implementation
                #print("Punish for unproductive move")

        # Penalty for an invalid move.
        if not self.lastMoveValid:
            reward -= 5
            #print("Punish for invalid move")

        # Reward for passengers served.
        reward += self.pplServedThisStep * 10
        

        # Update desired direction for next move 
        #if self.elevatorFloor < reqFloor:
        #    self.desiredDirection = 1   # Move up.
        #elif self.elevatorFloor > reqFloor:
        #    self.desiredDirection = -1  # Move down.
        #else:
        #    self.desiredDirection = 0   # Stay on the current floor.
        self.desiredDirection = self.deterministicControl()


        return reward

    #def returnStateComplex(self):
    #    # Flatten building state
    #    flatBuilding = self.building.flatten()
    #    
    #    # Robust normalization
    #    max_building_value = max(flatBuilding)
    #    flatBuildingNormalized = (flatBuilding / (max_building_value + 1e-8)) if max_building_value > 0 else flatBuilding
    #    
    #    # Normalize elevator floor
    #    normalized_floor = (self.elevatorFloor - 1) / (self.floors - 1)
    #    
    #    # One-hot encode last policy (already done)
    #    policy_encoding = np.eye(3)[self.lastPolicy+1]
    #    
    #    # Normalize car calls
    #    normalized_car_calls = [x / self.floors for x in self.carCalls]
    #    
    #    # Combine all normalized components
    #    output = np.concatenate([
    #        flatBuildingNormalized,
    #        [normalized_floor],
    #        policy_encoding,
    #        normalized_car_calls
    #    ])
    #    
    #    return output

    #Non-normalized state (complex)
    #def returnState(self):
    #    flatBuilding = self.building.flatten()
    #    output =  np.append([],flatBuilding)
    #    output = np.append(output,[self.elevatorFloor])
    #    output = np.append(output,np.eye(3)[self.lastPolicy+1])
    #    output = np.append(output, self.carCalls)
    #    return output


    ##FOR FUTURE: INCLUDE BOTH THE MAX WAIT TIME SQUARE (NORMALIZED) AND NUMBER OF PEOPLE WAIITNG (NORMALIZED) IN INPUT VECTOR
    def returnState(self):
        # Normalized floor requests (up/down)
        up_requests = self.buildingTotals[0] / self.capacity  # Shape: (10,)
        down_requests = self.buildingTotals[1] / self.capacity  # Shape: (10,)
        
        # Normalized elevator floor (1-10 → 0-1)
        elevator_floor = (self.elevatorFloor - 1) / (self.floors - 1)
        
        # Normalized car calls (0-10 → 0-1)
        car_calls = [x / self.floors for x in self.carCalls]

        #Previous policy
        policy_encoding = np.eye(3)[self.lastPolicy+1]

        current_floor_up = self.buildingTotals[0][self.elevatorFloor-1] / self.capacity
        current_floor_down = self.buildingTotals[1][self.elevatorFloor-1] / self.capacity
        total_waiting = np.sum(self.buildingTotals) / (2*self.floors*self.capacity)
        max_wait_time = np.max(self.building) / 100  # Normalized
        
        # Concatenate all components
        state = np.concatenate([
        up_requests, down_requests, [elevator_floor], 
        car_calls, policy_encoding,
        [total_waiting, max_wait_time],
        [current_floor_up, current_floor_down]  # New
        ])
        return state

    def __len__(self):
        return len(self.returnState())
    
    def deterministicControl(self):
        #First, see if anyone in the car would like to exit, return 0 if so
        for x in self.carCalls:
            if(x == self.elevatorFloor):
                #print("Someone exiting elevator")
                return 0
            
        #Next, see if anyone on the current floor wants to go in the elevator's direction, return 0 if so
        if(self.lastPolicy == -1):
            if(self.buildingTotals[1][self.elevatorFloor-1] >= 1):
                #print("someone entering elevator")
                return 0
        elif(self.lastPolicy == 1):
            if(self.buildingTotals[0][self.elevatorFloor-1] >= 1):
                #print("someone entering elevator")
                return 0
            
        #Now, we check if there are any current car calls. If there are, move in their direction.
        for x in self.carCalls:
            if(x != 0):
                if(x > self.elevatorFloor):
                    #print("move in car call direction")
                    return 1
                else:
                    #print("move in car call direction")
                    return -1
        
        
        
        #At this point, the elevator has no car requests and no requests on its current floor

        #See if there is a request present in the direction of previous travel, if there is, go there; else.
        if(self.lastPolicy == 1): #or (self.lastPolicy == 0 and self.lastLastPolicy == 1)):
            for floors in range(self.elevatorFloor + 1, self.floors + 1):
                if(self.buildingTotals[0][floors-1] >= 1 or self.buildingTotals[1][floors-1] >= 1):
                    #print("continue last direction")
                    return 1
        elif(self.lastPolicy == -1): #or (self.lastPolicy == 0 and self.lastLastPolicy == -1)):
            for floors in range(1, self.elevatorFloor):
                if(self.buildingTotals[0][floors-1] >= 1 or self.buildingTotals[1][floors-1] >= 1):
                    #print("continue last direction")
                    return -1
                
        #Check for boarding a passanger w/ a direction switch
        if(self.lastPolicy == 0):
            if(self.buildingTotals[1][self.elevatorFloor-1] >= 1 or self.buildingTotals[0][self.elevatorFloor-1] >= 1):
                return 0
                
        #If last policy was 0, go towards furthest request:
        if(self.lastPolicy == 0):
            self.buildingTotals

            # Convert to numpy array for easier manipulation (if not already)
            building_totals = np.array(self.buildingTotals)
    
            # Find columns where any value is greater than 0
            positive_floors = np.any(building_totals > 0, axis=0)
    
            # Get indices of such columns
            floor_indices = np.where(positive_floors)[0] + 1

            #If there are no requests return 0
            if(len(floor_indices) == 0 ): 
                return 0
            
            #Loop to find the furthest floor
            furthestFloor = floor_indices[0]
            for x in floor_indices:
                if(abs(self.elevatorFloor - furthestFloor) < abs(x - self.elevatorFloor)):
                    furthestFloor = x
            if(furthestFloor > self.elevatorFloor):
                return 1
            elif(furthestFloor < self.elevatorFloor):
                return -1
    

        #If all conditions are not met, the elevator must first stop to change directions
        return 0
    

    def greedyControl(self):
        #First, see if anyone in the car would like to exit, return 0 if so
        for x in self.carCalls:
            if(x == self.elevatorFloor):
                #print("Someone exiting elevator")
                return 0
            
        #Next, see if anyone on the current floor wants to go in the elevator's direction, return 0 if so
        if(self.lastPolicy == -1):
            if(self.buildingTotals[1][self.elevatorFloor-1] >= 1):
                #print("someone entering elevator")
                return 0
        elif(self.lastPolicy == 1):
            if(self.buildingTotals[0][self.elevatorFloor-1] >= 1):
                #print("someone entering elevator")
                return 0
            
        #Now, we check if there are any current car calls. If there are, move in their direction.
        for x in self.carCalls:
            if(x != 0):
                if(x > self.elevatorFloor):
                    #print("move in car call direction")
                    return 1
                else:
                    #print("move in car call direction")
                    return -1
        
        
        #Check for boarding a passanger w/ a direction switch
        if(self.lastPolicy == 0):
            if(self.buildingTotals[1][self.elevatorFloor-1] >= 1 or self.buildingTotals[0][self.elevatorFloor-1] >= 1):
                return 0

        #At this point, the elevator has no car requests and no requests on its current floor

        #See if there is a request present in the direction of previous travel, if there is, go there; else.
        if(self.lastPolicy == 1): #or (self.lastPolicy == 0 and self.lastLastPolicy == 1)):
            for floors in range(self.elevatorFloor + 1, self.floors + 1):
                if(self.buildingTotals[0][floors-1] >= 1 or self.buildingTotals[1][floors-1] >= 1):
                    #print("continue last direction")
                    return 1
        elif(self.lastPolicy == -1): #or (self.lastPolicy == 0 and self.lastLastPolicy == -1)):
            for floors in range(1, self.elevatorFloor):
                if(self.buildingTotals[0][floors-1] >= 1 or self.buildingTotals[1][floors-1] >= 1):
                    #print("continue last direction")
                    return -1
                
        
                
        #If last policy was 0, go towards furthest request:
        if(self.lastPolicy == 0):
            self.buildingTotals

            # Convert to numpy array for easier manipulation (if not already)
            building_totals = np.array(self.buildingTotals)
    
            # Find columns where any value is greater than 0
            positive_floors = np.any(building_totals > 0, axis=0)
    
            # Get indices of such columns
            floor_indices = np.where(positive_floors)[0] + 1

            #If there are no requests return 0
            if(len(floor_indices) == 0 ): 
                return 0
            
            #Loop to find the furthest floor
            closestFloor = floor_indices[0]
            for x in floor_indices:
                if(abs(self.elevatorFloor - closestFloor) < abs(x - self.elevatorFloor)):
                    closestFloor = x
            if(closestFloor > self.elevatorFloor):
                return 1
            elif(closestFloor < self.elevatorFloor):
                return -1
    

        #If all conditions are not met, the elevator must first stop to change directions
        return 0
    
    def getCurrentUsers(self):
        return self.currentUsers
        
        

        
            

