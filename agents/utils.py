import numpy as np
from gym.spaces import discrete

class Policy:

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def sample(self, state):

        raise NotImplementedError

class TabularPolicy(Policy):

    def __init__(self, state_space, action_space):
        if not isinstance(state_space, discrete.Discrete):
            raise TypeError("the state space must be of type {}".format(discrete.Discrete))
        if not isinstance(action_space, discrete.Discrete):
            raise TypeError("the action space must be of type {}".format(discrete.Discrete))

        super(TabularPolicy, self).__init__(state_space, action_space)

    def sample(self, state):
        pass

def display_grid_policy(policy, shape, action_symbols = None):
    ncol = shape[1]
    col = 0
    for state in


    for act_probs in policy:
        i = np.argmax(act_probs)
        print(action_symbols[i], end="")
        col += 1
        if col >= ncol:
            col = 0
            print()


def display_values(values):
    for i, val in enumerate(values):
        print(val, end=" ")
        if (i + 1) % 6 == 0:
            print()