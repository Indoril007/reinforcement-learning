import gym
import envs
import numpy as np
from agents.core import GridAgent
from agents.dynamic_methods import DynamicMethods
# from agents.dynamic_methods import policy_evaluation, policy_improvement, policy_iteration, value_iteration
from agents.monte_carlo import MonteCarlo

env = gym.make('SimpleGridWorld-v0')
agent = GridAgent(env.observation_space, env.action_space, env.shape, discount_factor = 0.99)
DM = DynamicMethods(env, agent)
DM.value_iteration()

agent.display_values()
agent.display_policy()
# env = gym.make('FrozenLake-v0').env

# transitions = env.P

# policy = [[0.25, 0.25, 0.25, 0.25] for _ in range(len(transitions))]
# optimal_policy = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
#                   [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0],
#                   [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
#                   [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]]

# for _ in range(10):
#     print(env.render())
#     env.step(env.action_space.sample())

# policy = TabularPolicy(env.observation_space, env.action_space)
# print("="*100)
# values = policy_evaluation(transitions, policy, max_iterations = 1000, end_states = [5])
# print("="*100)
#values = policy_evaluation(transitions, optimal_policy, max_iterations = 100, end_states = [5], discount_factor=0.99)

# print("="*100)
# policy = policy_iteration(transitions, end_states = [15])
# # print(policy)
# print("="*100)

# values = value_iteration(transitions, [5])
# MC = MonteCarlo(env, policy = [[0.25 for _ in range(4)] for _ in range(36)])
# MC = MonteCarlo(env, policy = policy)
# mc_values = MC._prediction(max_steps=1000, max_episodes=3)
# print("="*100)
# mc_values = MC._prediction(type='every', max_steps=2000, max_episodes=1000)
# print("="*100)
# mc_values = MC._prediction(max_steps=2000, max_episodes=1000)
# print("="*100)
# mc_values = MC._q_prediction()
# policy = MC.policy_iteration()


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
