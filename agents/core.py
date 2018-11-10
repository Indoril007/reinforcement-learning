from gym.spaces import discrete

class GridAgent(object):
    pass

class GridPolicy(object):
    pass

class GridValues(object):
    pass

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.solver = None

        if isinstance(env.observation_space, discrete.Discrete) and \
            isinstance(env.action_space, discrete.Discrete):
            self.values = TabularValues(env.observation_space, env.action_space)
            self.policy = TabularPolicy(env.observation_space, env.action_space)
        else:
            self.policy = None
            self.values = None

class TabularAgent(object):
    pass


class Values(object):

    def get_value(self, state):
        raise NotImplementedError

    def get_q_value(self, state, action):
        raise NotImplementedError

    def update_values(self, *args):
        raise NotImplementedError

    def update_q_values(self, *args):
        raise NotImplementedError

class TabularValues(Values):

    def __init__(self, num_states, num_actions, states = None, actions = None):
        self.values = [0]*num_states
        self.q_values = [[0] * num_actions for _ in range(num_states)]

    def get_value(self, state):
        return self.values[state]

    def get_q_value(self, state, action):
        return self.values[state][action]

    def update_values(self, state, value):
        self.values[state] = values

    def update_q_values(self, state, action, value):
        self.values[state][action] = value

class ValueFunction(Values):
    pass

class Policy:

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def sample_action(self, state):

        raise NotImplementedError

    def action_probs(self, state):

        raise NotImplementedError

    def optimal_action(self, state):

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

        self.state_names = state_names
        self.action_names = action_names
        self.states2index = None if state_names is None else dict(zip(state_names, range(len(state_names))))
        self.actions2index = None if action_names is None else dict(zip(action_names, range(len(action_names))))
        policy = [ [(1/action_space.n) for _ in range(action_space.n)] for _ in range(state_space.n) ]

    def sample_action(self, state):
        return np.random.choice(range(len(policy[state])) , p=policy[state])

    def sample_action_by_name(self, state):
        return np.random.choice(self.action_names, p=policy[self.states2index[state]])

    def action_probs(self, state):
        return policy[state]

    def action_probs_by_name(self, state):
        return policy[self.states2index[state]]

    def optimal_action(self, state):
        return np.argmax(self.action_probs(state))

    def optimal_action_by_name(self, state):
        return state_names[ np.argmax(self.action_probs_by_name(state)) ]

class GridPolicy(TabularPolicy):

    def __init__(self, state_space, action_space, shape):
        if not (shape[0] * shape[1]) == state_space.n:
            raise ValueError("nRows x nCols given by shape does not much the size of the state space")

        self.shape = shape
        self.state_names = [(i,j) for i in range(shape[0]) for j in range(shape[1])]
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

        super(GridPolicy, self).__init__(state_space, action_space, self.state_names, self.action_names)

    def display_policy(self):
        action_symbols = ['^', 'v', '<', '>']

        for state in self.state_names:
            i, j = state
            best_action = np.argmax(self.action_probs_by_name(state))
            if j == 0:
                print()
            print(action_symbols[best_action], end = "")


class Solver:
    pass