import numpy as np

def policy_evaluation(transitions, policy, end_states = None, discount_factor = 0.99, stop_threshold=0.001, max_iterations=None):
    num_states = len(transitions)
    values = np.zeros(num_states)
    iterations = 0

    while True:
        max_diff = 0
        for i in range(num_states):
            if i in end_states:
                continue
            new_value = 0

            for j, action_prob in enumerate(policy[i]):

                sum = 0
                for transition in transitions[i][j]:
                    prob, next_state, reward, done = transition
                    values[next_state] = 0 if done else values[next_state]
                    sum += prob * (reward + discount_factor*values[next_state])

                new_value += action_prob*sum

            max_diff = max(max_diff, abs(values[i] - new_value))
            values[i] = new_value

        print(values)
        iterations += 1
        if (max_iterations is not None and iterations >= max_iterations) or (max_diff < stop_threshold):
            break

    return values

def policy_improvement(transitions, values, discount_factor = 0.99):

    policy = [[0, 0, 0, 0] for _ in  range(len(values))]
    for i, state in enumerate(transitions):
        state = transitions[state]
        argmax = 0
        maxval = 0
        for j, action in enumerate(state):
            action = state[action]
            action_value = 0

            for transition in action:
                prob, next_state, reward, done = transition
                value = values[next_state]
                action_value += prob * (reward + discount_factor*value)

            if action_value > maxval:
                maxval = action_value
                argmax = j

        # print(policy)
        # print(i, argmax)
        policy[i][argmax] = 1

    return policy


def policy_iteration(transitions):
    pass

def value_iteration(transitions):
    pass