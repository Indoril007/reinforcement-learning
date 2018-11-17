import numpy as np

class TabularAgent(object):
    """
    The tabular agent class uses both tabular values and a tabular policy.
    This is in contrast to agents that use function approximation for values and/or policy
    """

    def __init__(self, nS: int, nA: int, discount: float) -> None:
        """
        :param nS: number of states in state space
        :param nA: number of actions in action space
        :param discount: the discount factor to be used in calculating return
        """

        self.nS = nS
        self.nA = nA
        self.discount = discount
        self.values = np.zeros(shape=nS)
        self.q_values = np.zeros(shape=(nS, nA))
        self.policy = TabularPolicy(nS, nA)

class TabularPolicy(object):

    def __init__(self, nS: int, nA: int) -> None:
        """
        :param nS: number of states in state space
        :param nA: number of actions in action space
        """

        self.nS = nS
        self.nA = nA
        self.policy = np.full(shape=(nS, nA), fill_value=1/nA)

    def sample(self, state: int) -> int:
        """
        Samples an action at a particular state according to probabilities given in the policy
        :param state: The state value
        :return: The sampled action
        """
        return np.random.choice(range(self.nA), p=self.policy[state])

    def set(self, state: int, action: int, epsilon: float = 0) -> bool:
        """
        Sets the epsilon-greedy action in a particular state. When epsilon is 0 this is completely greedy
        :rtype: bool: A boolean indicating whether the original policies values have been changed
        :param state: The state value
        :param action: The action value
        :param epsilon: The probability of taking a random action
        """
        probs = np.full(shape=self.nA, fill_value=epsilon/self.nA)
        probs[action] += 1 - epsilon

        if np.all(np.isclose(probs, self.policy[state])):
            return False

        np.copyto(self.policy[state], probs)
        return True

    def __getitem__(self, item):
        return self.policy[item]

    def __iter__(self):
        return self.policy.__iter__()
