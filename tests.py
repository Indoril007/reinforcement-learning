import gym
import envs
from agents.dynamic_methods import policy_evaluation, policy_improvement

env = gym.make('SimpleGridWorld-v0')

transitions = env.P
# policy = [[0.25, 0.25, 0.25, 0.25] for _ in range(len(transitions))]
optimal_policy = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                  [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0],
                  [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                  [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]]

# for _ in range(10):
#     print(env.render())
#     env.step(env.action_space.sample())

values = policy_evaluation(transitions, optimal_policy, max_iterations = 50, end_states = [5])

# for _ in range(10):
#     values = policy_evaluation(transitions, policy, max_iterations = 10, end_states = [5])
#     print(values)
#     policy = policy_improvement(transitions, values)
#     print(policy)
