import numpy as np
import matplotlib.pyplot as plt
from .core import Greedy, EpsilonGreedy

class TemporalDifference:

    def __init__(self, env, agent, log=True):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        self.agent = agent
        self.policy = agent.policy
        self.greedy = Greedy(agent.values)
        self.epsilon_greedy = EpsilonGreedy(agent.values)
        self.discount_factor = agent.discount_factor

        self.log = log


        self.fig = plt.figure()

    def policy_evaluation(self, alpha = 0.01, max_steps = 5000, max_episodes=1000, true_values = None):
        V = lambda x:self.agent.get_value(int(x))
        gamma = self.discount_factor

        for e in range(max_episodes):
            if self.log and e % 100 == 0:
                print("episode: {}".format(e))
                if not true_values is None:
                    print("error: {}".format(np.linalg.norm(true_values - np.array(self.agent.values.get_all_values()))))
            state = self.env.reset()

            for t in range(max_steps):
                action = self.policy.sample_action(state)
                next_state, reward, done, _ = self.env.step(action)
                new_value = V(state) + alpha*(reward + gamma*V(next_state) - V(state))
                self.agent.set_value(int(state), new_value)
                state = next_state
                if done:
                    break

    def sarsa(self, alpha = 0.01, max_steps = 1000000, max_episodes=1000, true_values = None, plot = False):
        Q = lambda s,a: self.agent.get_q_value(int(s), int(a))
        gamma = self.discount_factor
        behaviour = self.epsilon_greedy
        stats = {"ep_lengths":[], "returns": []}
        plt.interactive(False)
        len_means = []
        ret_means = []
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        for e in range(max_episodes):
            if self.log and e % 100 == 0:
                print("episode: {}".format(e))
                if not true_values is None:
                    print("error: {}".format(np.linalg.norm(true_values - np.array(self.agent.values.get_all_q_values()))))
                if behaviour.epsilon > 0.01:
                    behaviour.set_epsilon(behaviour.epsilon*0.99)
                len_means.append(np.mean(stats["ep_lengths"]))
                ret_means.append(np.mean(stats["returns"]))
                ax1.clear()
                ax2.clear()
                ax1.plot(len_means)
                ax2.plot(ret_means)
                stats["ep_lengths"] = []
                stats["returns"] = []
                self.fig.canvas.draw()
                plt.pause(0.001)
                behaviour.set_epsilon(behaviour.epsilon*0.99)
            # if e % 500 == 0:
            #     # behaviour.set_epsilon(behaviour.epsilon*0.99)
            #     # alpha *= 0.99
            #     self.agent.values.display_q_values()

            state = self.env.reset()
            action = behaviour.sample_action(state)
            G = 0
            for t in range(max_steps):
                # self.agent.values.display_q_values()
                # print(state, action)
                next_state, reward, done, _ = self.env.step(action)
                G += (gamma**t) * reward
                next_action = behaviour.sample_action(state)
                new_value = Q(state,action) + alpha*(reward + gamma*Q(next_state, next_action) - Q(state, action))
                self.agent.set_q_value(int(state), int(action), new_value)
                state = next_state
                action = next_action
                if done:
                    break

            stats["ep_lengths"].append(t)
            stats["returns"].append(G)


