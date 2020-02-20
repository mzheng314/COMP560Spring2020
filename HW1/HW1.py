import sys
import copy
import random
import time

BACKTRACK_COUNT = 0
LOCAL_COUNT = 0

colors = set()

class csp:
        
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        #represents domains available for backtrack search
        #represents assignments for local search
        self.domains = domains
        self.constraints = constraints
    
    def __str__(self):
        output=""
        for var in self.variables:
            value=self.domains[var]
            output+=var+" "+str(value)
            output+="\n"
        return(output)

#Backtrack Search

def AC3 (csp, queue=None):
     
    def arc_reduce(x,y):
        removals=[]
        change=False
        for vx in csp.domains[x].copy():
            found=False
            for vy in csp.domains[y]:
                if vx != vy:
                    found=True
            if(not found):
                csp.domains[x].remove(vx)
                removals.append((x,vx))
                change=True
        return change,removals
    
    removals=[]
    if queue is None:
        queue=[]
        for x in csp.variables:
            queue = queue + [(x, y) for y in csp.constraints[x]]
    while queue:
        x,y=queue.pop()
        c,r=arc_reduce(x,y)
        if r:
            removals.extend(r)
        if c:
            #not arc consistent
            if(len(csp.domains[x])==0):
                return False, removals
            #if we remove a value, check all neighbours
            else:
                queue = queue + [(x, z) for z in csp.constraints[x] if z!=y]

    return True, removals


def readCSPFromFile(pathToFile):
    if len(sys.argv) != 2:
        print("Invalid parameters. Expected 1 parameter (input file name)")
        exit()
    input_file = open(sys.argv[1], "r")
    input_by_line = input_file.readlines()
    #variable to track what sections of input we are in
    #0 for color input, 1 for node input, 2 for edge input
    input_state = 0
    #list of all colors
    global colors
    #dictionary containing each arc as 2 key-value pairs, one per direction
    constraints = {}
    for l in input_by_line:
        l=l.strip()
        if not l: #empty line is false
            input_state += 1 #move to next state
        elif input_state == 0:
            colors.add(l) 
        elif input_state == 1:
            constraints[l] = []
        elif input_state == 2:
            #add both key-value pairs for arc
            arc_nodes = l.split()
            constraints[arc_nodes[0]].append(arc_nodes[1]) 
            constraints[arc_nodes[1]].append(arc_nodes[0])
    domains = {}
    for node in constraints.keys(): #keys represent all the nodes
        domains[node] = colors #set all domains to all colors
    return csp(list(constraints.keys()),domains,constraints)

#finds next variable to use for search
#prioritizes adjacent unassigned, then picks pseudo-randomly
def findAdjacentUnassignedVariable(csp, assigned, past_var):
    #search for adjacent unassigned
    for var in csp.constraints[past_var]:
        if var not in assigned:
            return var
    #if no adjacent unassigned, search for others
    for var in [x for x in csp.variables if x not in csp.constraints[past_var]]:
        if var not in assigned:
            return var

#turns python set into iterable list
def domainSetToList(csp, var):
    values = [val for val in csp.domains[var]] 
    return values

def backTrackSearch(csp): #returns a solution or failure
    solved, assignment = backtrack({},csp,csp.variables[0])
    return solved, assignment

def backtrack(assignment, csp, var): #returns a solution or failure
    #var is variable currently being tested
    if len(assignment) == len(csp.variables):
        return True, assignment
    global BACKTRACK_COUNT
    BACKTRACK_COUNT += 1
    for value in domainSetToList(csp, var):
        value_works = True
        for con in csp.constraints[var]:
            #if conflict exists
            if (con in assignment.keys()) and (assignment[con] == value):
                value_works = False
                break
        if not value_works:
            continue
        else:
            assignment[var] = value
            next_var = findAdjacentUnassignedVariable(csp, assignment, var)
            #after each assignment, propagate changes with AC3
            if (not AC3(csp)):
                continue
            solved, csp = backtrack(assignment, csp, next_var)
            if solved:
                return True, assignment
            else:
                del assignment[var]
    return False, csp


#Local Search

#finds the number of current conflicts for a given var
def numConflicts(csp, var):
    num_cons = 0
    for con in csp.constraints[var]:
        if csp.domains[var] == csp.domains[con]:
            num_cons += 1
    return num_cons

def numTotalConflicts(csp):
    total = 0
    for v in csp.variables:
        total += numConflicts(csp, v)
    return total

def findMostConstrained(csp):
    most_constrained = ""
    highest_num_constraints = 0
    for v in csp.variables:
        if numConflicts(csp, v) > highest_num_constraints:
            most_constrained = v
            highest_num_constraints = numConflicts(csp, v)
    return most_constrained

def hillClimb(csp, var):
    #colors other than currently assigned color
    other_colors = [c for c in colors if c != csp.domains[var]]
    orig_color = csp.domains[var]
    least_constrained_color = csp.domains[var]
    num_conflicts = numConflicts(csp, var)
    for color in other_colors:
        csp.domains[var] = color
        if numConflicts(csp, var) < num_conflicts:
            least_constrained_color = color
            num_conflicts = numConflicts(csp, var)
    if least_constrained_color == orig_color:
        return False #unable to improve
    csp.domains[var] = least_constrained_color
    global LOCAL_COUNT
    LOCAL_COUNT+=1
    return True

def localSearch(csp, original_domains):
    start_time = time.time()
    #until 1 minute passes
    while (time.time() - start_time < 60):
        var = random.choice(csp.variables)
        while(hillClimb(csp,var) and numTotalConflicts(csp)!=0):
            var = findMostConstrained(csp)
            if(time.time() - start_time > 60):
                return False, csp
        if numTotalConflicts(csp) == 0:
            return True, csp
        else:
            csp.domains = original_domains
    return False, csp
    

def main():
    if len(sys.argv) != 2:
        print("Invalid parameters. Expected 1 parameter (input file name)")
        exit()
    input_file = open(sys.argv[1], "r")
    BackMap = readCSPFromFile(input_file)
    LocalMap = copy.deepcopy(BackMap)
    
    print("Backtrack search with AC3...")
        
    solution, assignment = backTrackSearch(BackMap)
    if(solution):
        for val in assignment:
            BackMap.domains[val] = assignment[val]
        print("Solution found by Backtrack Search: ")
        print(BackMap)
        print("Number of steps for Backtrack Search: {}".format(BACKTRACK_COUNT))
    else:
        print("Backtrack Search unable to find a solution.")

    print("Random Restart Local Search...")
    random_map_assign = {}
    #randomly assign colors to map
    for var in LocalMap.variables:
        random_map_assign[var] = random.sample(colors, 1)[0]
    LocalMap.domains = random_map_assign
    solution, csp = localSearch(LocalMap, random_map_assign)
    if(solution):
        print("Solution found by Local Search: ")
        print(LocalMap)
        print("Number of steps for Local Search: {}".format(LOCAL_COUNT))
    else:
        print("Local Search time exceeded one minute.")
    
if __name__ == "__main__":
    main()
