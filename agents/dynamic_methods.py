import numpy as np

def policy_evaluation(state_transitions, policy, end_states = None, discount_factor = 0.99, stop_threshold=0.001, max_iterations=None):
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
        print(values)

    return values

def _get_state_value(action_probs, action_transitions, values, discount_factor):
    value = 0

    for i, action_prob in enumerate(action_probs):

        action_value = _get_action_value(action_transitions[i], values, discount_factor)
        value += action_prob*action_value

    return value

def _get_action_value(transitions, values, discount_factor):
    action_value = 0
    for transition in transitions:
        prob, next_state, reward, done = transition
        values[next_state] = 0 if done else values[next_state]
        action_value += prob * (reward + discount_factor*values[next_state])
    return action_value

def policy_improvement(state_transitions, values, discount_factor = 0.99):

    policy = [[0, 0, 0, 0] for _ in  range(len(values))]
    for state in state_transitions:
        action_transitions = state_transitions[state]
        argmax = 0
        maxval = 0
        for action in action_transitions:
            transitions = action_transitions[action]
            action_value = 0

            action_value = _get_action_value(transitions, values, discount_factor)

            if action_value > maxval:
                maxval = action_value
                argmax = action

        policy[state][argmax] = 1

    return policy


def policy_iteration(transitions):


def value_iteration(transitions):
    pass