import numpy as np

class TemporalDifference:

    def __init__(self, env, agent, log=True):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        self.agent = agent
        self.policy = agent.policy
        self.discount_factor = agent.discount_factor

        self.log = log

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

    def sarsa(self, alpha = 0.01, max_steps = 5000, max_episodes=1000):
        Q = self.agent.get_q_value
        gamma = self.discount_factor

        for e in range(max_episodes):
            if self.log and e % 100 == 0:
                print("episode: {}".format(e))
                if not true_values is None:
                    print("error: {}".format(np.linalg.norm(true_values - np.array(self.agent.values.get_all_q_values()))))
            state = self.env.reset()
            action = self.policy.epsilon_greedy.sample_action(state)
            for t in range(max_steps):
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.policy.epsilon_greedy.sample_action(state)
                new_value = Q(state,action) + alpha*(reward + gamma*Q(next_state, next_action) - Q(state, action))
                self.agent.set_q_value(int(state), int(state), new_value)
                state = next_state
                action = next_action
                if done:
                    break


