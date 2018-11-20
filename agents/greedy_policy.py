import numpy as np

class GreedyPolicy(object):

    def __init__(self, q_values, epsilon=0):
        self.q_values = q_values
        self.nS, self.nA = q_values.shape
        self.epsilon = epsilon

    def probs(self, state):
        max_val = np.amax(self.q_values[state])
        optimal_actions = np.argwhere(np.isclose(self.q_values[state], max_val, rtol=0.001))
        probs = np.full(shape=self.nA, fill_value=self.epsilon/self.nA)
        probs[optimal_actions] += (1 - self.epsilon) / len(optimal_actions)
        return probs

    def sample(self, state):
        return np.random.choice(self.nA, p=self.probs(state))

    def get(self):
        probs = np.full(shape=(self.nS, self.nA), fill_value=self.epsilon/self.nA)
        for state in range(self.nS):
            max_val = np.amax(self.q_values[state])
            optimal_actions = np.argwhere(np.isclose(self.q_values[state], max_val, rtol=0.001))
            probs[state,optimal_actions] += (1-self.epsilon) / len(optimal_actions)
        return probs


