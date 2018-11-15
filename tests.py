import gym
import envs
import numpy as np
from agents.core import GridAgent
from agents.dynamic_methods import DynamicMethods
# from agents.dynamic_methods import policy_evaluation, policy_improvement, policy_iteration, value_iteration
from agents.monte_carlo import MonteCarlo
from agents.temporal_difference import TemporalDifference

env = gym.make('SimpleGridWorld-v0')
agent1 = GridAgent(env.observation_space, env.action_space, env.shape, discount_factor = 0.95)
agent2 = GridAgent(env.observation_space, env.action_space, env.shape, discount_factor = 0.95)
DM = DynamicMethods(env, agent1)
DM.policy_iteration()
DM.q_evaluation()

true_q_values = agent1.values.get_all_q_values()
true_values = agent1.values.get_all_values()
# agent1.display_values()
# agent1.display_policy()
# agent1.display_q_values()

# policy = agent1.policy.get_policy()
# agent2.policy.set_policy(policy)

MC = MonteCarlo(env, agent2)
MC.off_policy_q_iteration(max_episodes=10000, true_values=np.array(true_q_values))
MC.policy_improvement()

# TD = TemporalDifference(env, agent2)
# TD.policy_evaluation(true_values = true_values, max_episodes=3000, log=False)

agent2.display_values()
agent2.display_q_values()
agent2.display_policy()
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
