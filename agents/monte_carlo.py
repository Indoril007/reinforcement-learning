import numpy as np
from collections import defaultdict
from .core import Greedy, EpsilonGreedy


class MonteCarlo:

    def __init__(self, env, agent):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        self.agent = agent
        self.policy = agent.policy
        self.greedy = Greedy(self.agent.values)
        self.epsilon_greedy = EpsilonGreedy(self.agent.values)
        self.discount_factor = agent.discount_factor

    def value_prediction(self, style = 'first', max_steps = 1000, max_episodes=100):

        num_visits = defaultdict(int)

        for e in range(max_episodes):
            obs = self.env.reset()

            episode = []
            visits = defaultdict(list)
            for t in range(max_steps):
                action = self.policy.sample_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                episode.append([obs, action, reward])

                if (style == 'every') or (style == 'first' and not obs in visits):
                    visits[obs].append(t)
                    num_visits[obs] += 1

                if done:
                    break
                obs = next_obs

            ret = 0
            for step in episode[::-1]:
                _, _, reward = step
                ret = self.discount_factor * ret + reward
                step.append(ret)

            for obs in visits:
                visited_timesteps = visits[obs]
                n = len(visited_timesteps)
                sum_returns = sum([episode[t][3] for t in visited_timesteps])
                old_value = self.agent.get_value(int(obs))
                new_value = old_value + (1/num_visits[obs]) * (sum_returns - n*old_value)
                self.agent.set_value(int(obs), new_value)

    def q_prediction(self, style = 'first', max_steps = 1000, max_episodes=100):

        num_visits = defaultdict(lambda: defaultdict(int))

        for e in range(max_episodes):
            obs = self.env.reset()

            episode = []
            visits = defaultdict(lambda: defaultdict(list))
            for t in range(max_steps):
                action = self.policy.sample_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                episode.append([obs, action, reward])

                if (style == 'every') or (style == 'first' and not action in visits[obs]):
                    visits[obs][action].append(t)
                    num_visits[obs][action] += 1

                if done:
                    break
                obs = next_obs

            ret = 0
            for step in episode[::-1]:
                _, _, reward = step
                ret = self.discount_factor * ret + reward
                step.append(ret)

            for obs in visits:
                for action in visits[obs]:
                    visited_timesteps = visits[obs][action]
                    n = len(visited_timesteps)
                    sum_returns = sum([episode[t][3] for t in visited_timesteps])
                    old_value = self.agent.get_q_value(int(obs), action)
                    new_value = old_value + (1/num_visits[obs][action]) * (sum_returns - n*old_value)
                    self.agent.set_q_value(int(obs), action, new_value)

    def off_policy_q_prediction(self, max_steps = 1000, max_episodes=100, style='every'):
        target = self.greedy
        behaviour = self.epsilon_greedy

        cum_weights = defaultdict(lambda: defaultdict(int))

        for e in range(max_episodes):
            if e % 10 == 0:
                print("episode {}".format(e))
            obs = self.env.reset()

            episode = []
            visits = defaultdict(lambda: defaultdict(list))
            for t in range(max_steps):
                action = behaviour.sample_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                episode.append([obs, action, reward])

                if style == 'every':
                    visits[obs][action].append(t)

                if done:
                    break
                obs = next_obs

            ret = 0
            weight = 1
            for step in episode[::-1]:
                obs, action, reward = step
                ret = self.discount_factor * ret + reward
                cum_weights[obs][action] += weight
                old_value = self.agent.get_q_value(int(obs), action)
                new_value = old_value + (weight/cum_weights[obs][action]) * (ret - old_value)
                self.agent.set_q_value(int(obs), action, new_value)

                importance_ratio = (target.get_action_prob(obs, action) /
                                     behaviour.get_action_prob(obs, action))
                weight = weight * importance_ratio
                if weight == 0:
                    break

    def off_policy_q_iteration(self, max_steps = 5000, max_episodes=100, true_values=None):
        target = self.greedy
        behaviour = self.epsilon_greedy

        cum_weights = defaultdict(lambda: defaultdict(int))

        for e in range(max_episodes):
            if e % 100 == 0:
                print("episode {}".format(e))
                if not true_values is None:
                    print("error {}".format(np.linalg.norm(true_values - np.array(self.agent.values.get_all_q_values()))))
            obs = self.env.reset()


            episode = []
            visits = defaultdict(lambda: defaultdict(list))
            for t in range(max_steps):
                action = behaviour.sample_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                episode.append([obs, action, reward])
                visits[obs][action].append(t)
                if done:
                    break
                obs = next_obs

            ret = 0
            weight = 1
            for step in episode[::-1]:
                obs, action, reward = step
                ret = self.discount_factor * ret + reward
                cum_weights[obs][action] += weight
                old_value = self.agent.get_q_value(int(obs), action)
                new_value = old_value + (weight/cum_weights[obs][action]) * (ret - old_value)
                self.agent.set_q_value(int(obs), action, new_value)
                if action != target.sample_action(obs):
                    break
                importance_ratio = (1 / behaviour.get_action_prob(obs, action))
                weight = weight * importance_ratio

    def policy_improvement(self, type = 'greedy'):

        for state in range(self.num_states):
            optimal_action = None
            maxval = None

            for action in range(self.num_actions):
                val = self.agent.get_q_value(state, action)

                if maxval is None or val > maxval:
                    optimal_action = action
                    maxval = val

            if type == 'greedy':
                self.policy.set_optimal_action(state, optimal_action)
            elif type == 'epsilon_greedy':
                self.policy.set_optimal_action(state, optimal_action, epsilon = 0.1)

    def policy_iteration(self, style = 'first', max_steps = 1000, max_episodes=10, type='greedy'):
        for _ in range(10):
            print(_)
            self.q_prediction(style, max_steps, max_episodes)
            self.policy_improvement(type=type)

    def value_iteration(self, style = 'first', max_steps = 1000, type='optimal'):
        for _ in range(100):
            self.q_prediction(style, max_steps, max_episodes=1)
            self.policy_improvement(type=type)
