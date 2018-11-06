import gym
import envs
import numpy as np
from agents.dynamic_methods import policy_evaluation, policy_improvement, policy_iteration, value_iteration

env = gym.make('SimpleGridWorld-v0')
# env = gym.make('FrozenLake-v0').env

transitions = env.P

# policy = [[0.25, 0.25, 0.25, 0.25] for _ in range(len(transitions))]
optimal_policy = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                  [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0],
                  [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                  [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]]

# for _ in range(10):
#     print(env.render())
#     env.step(env.action_space.sample())


#values = policy_evaluation(transitions, [[0.25 for _ in range(4)] for _ in range(36)], max_iterations = 100, end_states = [5], discount_factor=0.9)
#values = policy_evaluation(transitions, optimal_policy, max_iterations = 100, end_states = [5], discount_factor=0.99)

# policy = policy_iteration(transitions, end_states = [15])
# print(policy)

values = value_iteration(transitions, [5])

# def _display_policy(policy):
#     col = 0
#     for act_probs in policy:
#         i = np.argmax(act_probs)
#         print(actions[i], end = "")
#         col += 1
#         if col >= ncol:
#             col = 0
#             print()

# actions = ['^', 'v', '<', '>']
# ncol = env.ncol
# _display_policy(policy)

# for _ in range(10):
#     values = policy_evaluation(transitions, policy, max_iterations = 10, end_states = [5])
#     print(values)
#     policy = policy_improvement(transitions, values)
#     print(policy)
