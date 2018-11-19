import numpy as np

class GreedyPolicy(object):

    def __init__(self, q_values, epsilon=0):
        self.q_values = q_values
        self.nS, self.nA = q_values.shape
        self.epsilon = epsilon

    def sample(self, state):
        optimal_action = np.argmax(self.q_values[state])
        probs = np.full(shape=self.nA, fill_value=self.epsilon/self.nA)
        probs[optimal_action] += 1 - self.epsilon
        return np.random.choice(self.nA, p=probs)

    def get(self):
        optimal_actions = np.argmax(self.q_values, axis=1)
        probs = np.full(shape=(self.nS, self.nA), fill_value=self.epsilon/self.nA)
        probs[np.arange(len(probs)), optimal_actions] += 1-self.epsilon
        return probs


