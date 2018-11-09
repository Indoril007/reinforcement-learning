import numpy as np
import gym
from collections import defaultdict

class MonteCarlo:

    def __init__(self, env, policy = None, discount = 0.99):
        self.env = env
        self.policy = policy
        self.discount = discount

    def _prediction(self, type = 'first', max_steps = 1000, max_episodes=100):

        num_visits = defaultdict(int)
        values = defaultdict(float)

        for e in range(max_episodes):
            obs = self.env.reset()

            episode = []
            visits = defaultdict(list)
            for t in range(max_steps):
                action = np.random.choice(len(self.policy[obs]), p = self.policy[obs])
                next_obs, reward, done, _ = self.env.step(action)
                episode.append([obs, action, reward])

                if (type == 'every') or (type == 'first' and not obs in visits):
                    visits[obs].append(t)
                    num_visits[obs] += 1

                if done:
                    break
                obs = next_obs

            ret = 0
            for step in episode[::-1]:
                _, _, reward = step
                ret = self.discount * ret + reward
                step.append(ret)

            for obs in visits:
                visited_timesteps = visits[obs]
                n = len(visited_timesteps)
                sum_returns = sum([episode[t][3] for t in visited_timesteps])
                values[obs] = values[obs] + (1/num_visits[obs]) * (sum_returns - n*values[obs])

        self._display_values(values)
        return values

    def _q_prediction(self, type = 'first', max_steps = 1000, max_episodes=100):

        num_visits = defaultdict(lambda: defaultdict(int))
        values = defaultdict(lambda: defaultdict(float))

        for e in range(max_episodes):
            obs = self.env.reset()

            episode = []
            visits = defaultdict(lambda: defaultdict(list))
            for t in range(max_steps):
                action = np.random.choice(len(self.policy[obs]), p = self.policy[obs])
                next_obs, reward, done, _ = self.env.step(action)
                episode.append([obs, action, reward])

                if (type == 'every') or (type == 'first' and not action in visits[obs]):
                    visits[obs][action].append(t)
                    num_visits[obs][action] += 1

                if done:
                    break
                obs = next_obs

            ret = 0
            for step in episode[::-1]:
                _, _, reward = step
                ret = self.discount * ret + reward
                step.append(ret)

            for obs in visits:
                for action in visits[obs]:
                    visited_timesteps = visits[obs][action]
                    n = len(visited_timesteps)
                    sum_returns = sum([episode[t][3] for t in visited_timesteps])
                    values[obs][action] = values[obs][action] + (1/num_visits[obs][action]) * (sum_returns - n*values[obs][action])

        return values

    def policy_improvement(self, q_values):

        for state in q_values:
            argmax = None
            maxval = None

            for action in q_values[state]:
                val = q_values[state][action]

                if maxval is None or val > maxval:
                    argmax = action
                    maxval = val

            for i, action in enumerate(self.policy[state]):
                self.policy[state][i] = 0

            self.policy[state][argmax] = 1

    def policy_iteration(self, type = 'first', max_steps = 1000, max_episodes=10):
        for _ in range(10):
            print(_)
            q_values = self._q_prediction(type, max_steps, max_episodes)
            self.policy_improvement(q_values)

        self._display_policy(self.policy)
        return self.policy



    def _display_values(self, values):
        for i in range(36):
            print(values[i] if i in values else 0, end=" ")
            if (i+1) % 6 == 0:
                print()

    def _display_policy(self, policy):
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

