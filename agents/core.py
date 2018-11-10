from gym.spaces import discrete
import numpy as np

class Agent(object):
    def __init__(self, state_space, action_space, discount_factor):
        self.state_space = state_space
        self.action_space = action_space
        self.discount_factor = discount_factor

        self.values = None
        self.policy = None

    def display_values(self):
        self.values.display()

    def display_policy(self):
        self.policy.display()


class TabularAgent(Agent):

    def __init__(self, state_space, action_space, discount_factor):
        super(TabularAgent, self).__init__(state_space, action_space, discount_factor)

class GridAgent(TabularAgent):

    def __init__(self, state_space, action_space, shape, discount_factor):
        super(GridAgent, self).__init__(state_space, action_space, discount_factor)

        self.policy = GridPolicy(state_space, action_space, shape)
        self.values = GridValues(state_space, action_space, shape)

        self.get_value = self.values.get_value
        self.get_q_value = self.values.get_q_value
        self.set_value = self.values.set_value
        self.set_q_value = self.values.set_q_value

class Values(object):

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def get_value(self, state):
        raise NotImplementedError

    def get_q_value(self, state, action):
        raise NotImplementedError

    def set_value(self, state, value):
        raise NotImplementedError("set value has not been defined for these values. Likely because they are not tabular")

    def set_q_value(self, state, action_value):
        raise NotImplementedError("set q_value has not been defined for these values. Likely because they are not tabular")

    def display(self):
        raise NotImplementedError

class TabularValues(Values):

    def __init__(self, state_space, action_space):
        if not isinstance(state_space, discrete.Discrete):
            raise TypeError("For tabular values the state space must be discrete")
        if not isinstance(action_space, discrete.Discrete):
            raise TypeError("For tabular values the action space must be discrete")

        super(TabularValues, self).__init__(state_space, action_space)

        self.num_states = state_space.n
        self.num_actions = action_space.n
        self.values = [0 for _ in range(self.num_states)]
        self.q_values = [[0 for _ in range(self.num_actions)] for _ in range(self.num_states)]

    def get_value(self, state):
        return self.values[state]

    def get_q_value(self, state, action):
        return self.q_values[state][action]

    def set_value(self, state, value):
        self.values[state] = value

    def set_q_value(self, state, action, value):
        self.q_values[state][action] = value

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __iter__(self):
        return self.values.__iter__()

class GridValues(TabularValues):

    def __init__(self, state_space, action_space, shape):

        super(GridValues, self).__init__(state_space, action_space)

        if not self.num_states == (shape[0] * shape[1]):
            raise ValueError("The num of columns and rows given in shape does not match up with the number of states in"
                             "the environment")

        self.shape = shape
        self.nRows = shape[0]
        self.nCols = shape[1]

    def _convert_to_state(self, pos):
        row, col = pos
        return row * self.nRows + col

    def get_value(self, state):
        if type(state) is int:
            return super(GridValues, self).get_value(state)
        elif type(state) is tuple:
            return super(GridValues, self).get_value(self._convert_to_state(state))

    def get_q_value(self, state, action):
        if type(state) is int:
            return super(GridValues, self).get_q_value(state, action)
        elif type(state) is tuple:
            return super(GridValues, self).get_q_value(self._convert_to_state(state), action)

    def set_value(self, state, value):
        if type(state) is int:
            super(GridValues, self).set_value(state, value)
        elif type(state) is tuple:
            super(GridValues, self).set_value(self._convert_to_state(state), value)

    def set_q_value(self, state, action, value):
        if type(state) is int:
            super(GridValues, self).set_q_value(state, action, value)
        elif type(state) is tuple:
            super(GridValues, self).set_q_value(self._convert_to_state(state), action, value)

    def display(self):
        for r in range(0, self.num_states, self.nCols):
            print(" ".join(map(lambda x: "{:.4f}".format(x), self.values[r:r+self.nCols])))

class Policy:

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def sample_action(self, state):
        raise NotImplementedError

    def get_action_probs(self, state):
        raise NotImplementedError

    def get_optimal_action(self, state):
        raise NotImplementedError

    def set_optimal_action(self, state, action):
        raise NotImplementedError

    def display(self):
        raise NotImplementedError

class TabularPolicy(Policy):

    def __init__(self, state_space, action_space, state_names = None, action_names = None):
        if not isinstance(state_space, discrete.Discrete):
            raise TypeError("the state space must be of type {}".format(discrete.Discrete))
        if not isinstance(action_space, discrete.Discrete):
            raise TypeError("the action space must be of type {}".format(discrete.Discrete))
        if state_names is not None and len(state_names) != state_space.n:
            raise ValueError("The length of state_names should be equal to the size of the state space")
        if action_names is not None and len(action_names) != action_space.n:
            raise ValueError("The length of action_names should be equal to the size of the action space")

        super(TabularPolicy, self).__init__(state_space, action_space)

        self.num_states = state_space.n
        self.num_actions = action_space.n
        self.state_names = state_names
        self.action_names = action_names
        self.states2index = None if state_names is None else dict(zip(state_names, range(len(state_names))))
        self.actions2index = None if action_names is None else dict(zip(action_names, range(len(action_names))))
        self.policy = [ [(1/action_space.n) for _ in range(action_space.n)] for _ in range(state_space.n) ]

    def sample_action(self, state):
        return np.random.choice(range(len(self.policy[state])) , p=self.policy[state])

    def sample_action_by_name(self, state):
        return np.random.choice(self.action_names, p=self.policy[self.states2index[state]])

    def get_action_probs(self, state):
        return self.policy[state]

    def get_action_probs_by_name(self, state):
        return {a : self.policy[self.states2index[state]][self.actions2index[a]] for a in self.action_names}

    def get_optimal_action(self, state):
        return np.argmax(self.action_probs(state))

    def get_optimal_action_by_name(self, state):
        return state_names[ np.argmax(self.action_probs_by_name(state)) ]

    def set_optimal_action(self, state, optimal_action):
        if self.policy[state][optimal_action] == 1:
            return False

        for a in range(self.num_actions):
            self.policy[state][a] = 0
        self.policy[state][optimal_action] = 1
        return True

    def set_optimal_action_by_name(self, state, optimal_action):
        if self.policy[self.states2index[state]][self.actions2index[optimal_action]] == 1:
            return False

        for a in range(self.num_actions):
            self.policy[self.states2index[state]][a] = 0
        self.policy[self.states2index[state]][self.actions2index[optimal_action]] = 1
        return True

class GridPolicy(TabularPolicy):

    def __init__(self, state_space, action_space, shape):
        if not (shape[0] * shape[1]) == state_space.n:
            raise ValueError("nRows x nCols given by shape does not much the size of the state space")

        self.shape = shape
        self.state_names = [(i,j) for i in range(shape[0]) for j in range(shape[1])]
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

        super(GridPolicy, self).__init__(state_space, action_space, self.state_names, self.action_names)

    def display(self):
        action_symbols = ['^', 'v', '<', '>']

        for state in self.state_names:
            i, j = state
            best_action = np.argmax(self.get_action_probs(self.states2index[state]))
            if (not i == 0) and (j == 0):
                print()
            print(action_symbols[best_action], end = "")

        print()
