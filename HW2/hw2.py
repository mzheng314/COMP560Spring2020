import sys, copy, random
global true_probs
#true_probs format: {State: {Action: {Result: probability, Result: probability} } }
#States: Fairway, Ravine, Close, Same, Left, Over, In
#Actions: Fairway, Ravine - At, Past, Left
#         Over - Chip, Pitch
#         Same, Left, Close - Putt

def next_state(state, action):
    global true_probs
    rand_val = random.random()
    #adds probabilities until they are larger than rand_val, at which point that state is returned
    rand_sum = 0
    for result in true_probs[state][action]:
        rand_sum += true_probs[state][action][result]
        if rand_sum >= rand_val:
            return result

def model_based_solve(delta, epsilon):
    print("Model-based Reinforcement Learning:")
    global true_probs
    #create dictionary with same structure to hold count of the number of times each transition from state 1 with action goes to state 2
    trans_counts = copy.deepcopy(true_probs)
    #create dictionary to store utilities of states
    utils = {}
    #create dictionary to track number of iteration of action from given state
    total_iterations = {}
    for start_state in trans_counts:
        utils[start_state] = 0
        total_iterations[start_state] = {}
        for action in trans_counts[start_state]:
            total_iterations[start_state][action] = 0
            for end_state in trans_counts[start_state][action]:
                trans_counts[start_state][action][end_state] = 0
    utils["In"] = 1
    end_condition = False
    cur_state = "Fairway"
    max_change = 0 #maximum change in probability compared against epsilon for end condition
    while not end_condition:
        #determine next action
        possible_actions = [] #list of potential chosen actions based expected utility and/or exploration criteria
        min_utility = float("inf") 
        for action in trans_counts[cur_state]:
            action_count = 0 #number of times this action has been taken from this state
            expected_util = 0
            for end_state in trans_counts[cur_state][action]:
                action_count += trans_counts[cur_state][action][end_state]
                if total_iterations[cur_state][action] != 0:
                    expected_util += trans_counts[cur_state][action][end_state]/total_iterations[cur_state][action]*utils[end_state]
            if action_count < 30: #controls exploration; higher values enable more exploration
                expected_util = 2 #estimate of minimum utility to encourage exploration
                if expected_util < min_utility:
                    possible_actions = []
                    min_utility = expected_util
                possible_actions.append(action)
            elif expected_util <= min_utility:
                if expected_util < min_utility:
                    possible_actions = []
                    min_utility = expected_util
                possible_actions.append(action)
        #choose action and update counts
        chosen_action = random.choice(possible_actions)
        new_state = next_state(cur_state, chosen_action)
        old_prob = 0
        if total_iterations[cur_state][chosen_action] > 1:
            old_prob = (trans_counts[cur_state][chosen_action][new_state])/(total_iterations[cur_state][chosen_action])
        total_iterations[cur_state][chosen_action] += 1
        trans_counts[cur_state][chosen_action][new_state] += 1
        #update max_change
        new_prob = trans_counts[cur_state][chosen_action][new_state]/total_iterations[cur_state][chosen_action]
        max_change = max(max_change, new_prob - old_prob)
        #transtion to new state
        cur_state = new_state
        
        #update utility values
        new_utils = {}
        for start_state in utils:
            if start_state == "In":
                new_utils["In"] = 1
                continue
            #Bellman equation
            action_util_sum_list = []
            for action in trans_counts[start_state]:
                temp = 0
                for end_state in trans_counts[start_state][action]:
                    if total_iterations[start_state][action] != 0:
                        temp += trans_counts[start_state][action][end_state]/total_iterations[start_state][action]*utils[end_state]
                action_util_sum_list.append(temp)
            new_utils[start_state] = 1 + delta*min(action_util_sum_list) #reward is always 1 in golf
        utils = new_utils

        #reset if needed and check end condition
        if cur_state == "In":
            cur_state = "Fairway"
            if max_change < epsilon:
                end_condition = True
            else:
                max_change = 0
    out_f = open("output_model_based.txt", "w")
    for state in trans_counts:
        for action in trans_counts[state]:
            for end in trans_counts[state][action]:
                triplet = "{state}/{action}/{end}/{prob}\n".format(state = state, action = action, end = end, prob = (0 if total_iterations[state][action] == 0 else trans_counts[state][action][end]/total_iterations[state][action]))
                print(triplet, end="")
                out_f.write(triplet)
    out_f.close()

def model_free_solve(delta, epsilon):
    print("Model-free Reinforcement Learning:")
    global true_probs
    cur_state = "Fairway"
    end_condition = False
    utils = {}
    for state in true_probs:
        utils[state] = {}
        for action in true_probs[state]:
            utils[state][action] = 0
    transition_count = copy.deepcopy(utils)
    iteration_count = 0
    max_change = 0
    while not end_condition:
        possible_actions = [] #list of potential chosen actions based expected utility and/or exploration criteria
        min_utility = float("inf") 
        for action in utils[cur_state]:
            #find set of possible new actions, including all actions that are yet to be explored
            if transition_count[cur_state][action] < 30: #controls exploration; higher values enable more exploration
                min_utility = 0.5 #estimate of minimum utility to encourage exploration
                possible_actions.append(action)
            else:
                if utils[cur_state][action] < min_utility:
                    possible_actions = []
                    possible_actions.append(action)
                    min_utility = utils[cur_state][action]
                elif utils[cur_state][action] == min_utility:
                    possible_actions.append(action)
        chosen_action = random.choice(possible_actions)
        transition_count[cur_state][action] += 1
        new_state = next_state(cur_state, chosen_action)
        td_update_util_value = 0
        if new_state != "In":
            for action in utils[new_state]:
                if utils[new_state][action] > td_update_util_value:
                    td_update_util_value = utils[new_state][action]
        #TD update Q value for util
        utils[cur_state][chosen_action] += (60/(60+iteration_count))*(1+delta*td_update_util_value-utils[cur_state][chosen_action])
        max_change = max((60/(60+iteration_count))*(1+delta*td_update_util_value-utils[cur_state][chosen_action]), max_change)

        #change states
        cur_state = new_state

        #reset if necessary
        if cur_state == "In":
            cur_state = "Fairway"
            if max_change < epsilon:
                end_condition = True
            else:
                max_change = 0

        iteration_count += 1
    out_f = open("output_model_free.txt", "w")
    for state in utils:
        for action in utils[state]:
            triplet = "{state}/{action}/{util}\n".format(state = state, action = action, util=utils[state][action])
            print(triplet, end="")
            out_f.write(triplet)
    out_f.close()


if __name__ == "__main__":
    random.seed()
    f = open(str(sys.argv[1]), "r")
    #f = open("test.txt", "r")
    global true_probs
    true_probs = {}
    for line in f:
        prob_list = line.split("/")
        #ensure dictionary strcture is created
        if prob_list[0] not in true_probs:
            true_probs[prob_list[0]] = {}
        if prob_list[1] not in true_probs[prob_list[0]]:
            true_probs[prob_list[0]][prob_list[1]] = {}
        #add probability to true probability dictionary
        true_probs[prob_list[0]][prob_list[1]][prob_list[2]] = float(prob_list[3].strip())
    model_based_solve(0.9, 0.005)
    model_free_solve(0.9, 0.0005)
