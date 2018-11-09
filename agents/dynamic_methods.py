import numpy as np

def policy_evaluation(state_transitions, policy, end_states = None, discount_factor = 0.99, stop_threshold=0.00001, max_iterations=None):
    num_states = len(state_transitions)
    values = np.zeros(num_states)
    iterations = 0

    while True:
        max_diff = 0
        for i in range(num_states):
            if i in end_states:
                continue

            new_value = _get_state_value(policy[i], state_transitions[i], values, discount_factor)
            max_diff = max(max_diff, abs(values[i] - new_value))
            values[i] = new_value

        iterations += 1
        if (max_iterations is not None and iterations >= max_iterations) or (max_diff < stop_threshold):
            break

    _display_values(values)
    return values

def _get_state_value(action_probs, action_transitions, values, discount_factor):
    value = 0

    for i, action_prob in enumerate(action_probs):

        action_value = _get_action_value(action_transitions[i], values, discount_factor)
        value += action_prob*action_value

    return value

def _get_opt_state_value(action_transitions, values, discount_factor):
    maxval = None

    for action in action_transitions:
        transitions = action_transitions[action]
        action_value = _get_action_value(transitions, values, discount_factor)

        if maxval is None or action_value > maxval:
            maxval = action_value

    return maxval

def _get_action_value(transitions, values, discount_factor):
    action_value = 0
    for transition in transitions:
        prob, next_state, reward, done = transition
        values[next_state] = 0 if done else values[next_state]
        action_value += prob * (reward + discount_factor*values[next_state])
    return action_value

def policy_improvement(state_transitions, values, discount_factor = 0.99):
    nS = len(state_transitions)
    nA = len(state_transitions[0])

    policy = [[0 for _ in range(nA)] for _ in range(nS)]
    for state in state_transitions:
        action_transitions = state_transitions[state]
        argmax = 0
        maxval = None
        for action in action_transitions:
            transitions = action_transitions[action]
            action_value = 0

            action_value = _get_action_value(transitions, values, discount_factor)

            if maxval is None or action_value > maxval:
                maxval = action_value
                argmax = action

        policy[state][argmax] = 1

    return policy


def policy_iteration(state_transitions, end_states):
    nS = len(state_transitions)
    nA = len(state_transitions[0])

    # initial policy to be uniformly distributed across actions
    old_policy = None
    new_policy = [[(1/nA) for _ in range(nA)] for _ in range(nS)]
    num_iterations = 0

    while new_policy != old_policy:
        old_policy = new_policy
        values = policy_evaluation(state_transitions, old_policy, end_states = end_states, max_iterations=100)
        new_policy = policy_improvement(state_transitions, values)
        num_iterations += 1

    _display_policy(new_policy)
    _display_values(values)
    return new_policy

def value_iteration(state_transitions, end_states, stop_threshold = 0.01, max_iterations = None, discount_factor = 0.99):
    nS = len(state_transitions)
    nA = len(state_transitions[0])
    values = np.zeros(nS)
    iterations = 0

    while True:
        max_diff = 0
        for i in range(nS):
            if i in end_states:
                continue

            new_value = _get_opt_state_value(state_transitions[i], values, discount_factor)
            max_diff = max(max_diff, abs(values[i] - new_value))
            values[i] = new_value

        iterations += 1
        if (max_iterations is not None and iterations >= max_iterations) or (max_diff < stop_threshold):
            break

    _display_values(values)
    return values

def _display_policy(policy):
    actions = ['^', 'v', '<', '>']
    ncol = 6
    col = 0
    for act_probs in policy:
        i = np.argmax(act_probs)
        print(actions[i], end = "")
        col += 1
        if col >= ncol:
            col = 0
            print()

def _display_values(values):
    for i, val in enumerate(values):
        print(val, end=" ")
        if (i+1) % 6 == 0:
            print()

