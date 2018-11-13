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

    def display_q_values(self):
        self.values.display_q_values()

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
        self.get_q_values = self.values.get_q_values
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

    def get_all_q_values(self):
        raise NotImplementedError

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

    def get_q_values(self, state):
        return self.q_values[state]

    def get_all_q_values(self):
        return self.q_values

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
        else:
            raise TypeError("state should be an integer or tuple")

    def get_q_value(self, state, action):
        if type(state) is int:
            return super(GridValues, self).get_q_value(state, action)
        elif type(state) is tuple:
            return super(GridValues, self).get_q_value(self._convert_to_state(state), action)
        else:
            raise TypeError("state should be an integer or tuple")

    def set_value(self, state, value):
        if type(state) is int:
            super(GridValues, self).set_value(state, value)
        elif type(state) is tuple:
            super(GridValues, self).set_value(self._convert_to_state(state), value)
        else:
            raise TypeError("state should be an integer or tuple")

    def set_q_value(self, state, action, value):
        if type(state) is int:
            super(GridValues, self).set_q_value(state, action, value)
        elif type(state) is tuple:
            super(GridValues, self).set_q_value(self._convert_to_state(state), action, value)
        else:
            raise TypeError("state should be an integer or tuple")

    def display(self):
        for r in range(0, self.num_states, self.nCols):
            print(" ".join(map(lambda x: "{:.4f}".format(x), self.values[r:r+self.nCols])))

    def display_q_values(self):
        lines = ["|", "|", "|", "-"]
        for state in range(self.num_states):
            if state != 0 and state % self.nCols == 0:
                print("\n".join(lines))
                lines = ["|", "|", "|","-"]
            lines[0] += "{:^18.2f}|".format(self.q_values[state][0])
            lines[1] += " {:<8.2f}{:>8.2f} |".format(self.q_values[state][2], self.q_values[state][3])
            lines[2] += "{:^18.2f}|".format(self.q_values[state][1])
            lines[3] += "-"*19
        print("\n".join(lines))

class Policy:

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.greedy = Greedy(self)
        self.epsilon_greedy = EpsilonGreedy(self)

    def sample_action(self, state):
        raise NotImplementedError

    def get_action_probs(self, state):
        raise NotImplementedError

    def get_action_prob(self, state, action):
        raise NotImplementedError

    def get_policy(self):
        raise NotImplementedError

    def set_policy(self):
        raise NotImplementedError

    def display(self):
        raise NotImplementedError

class Greedy(object):

    def __init__(self, parent):
        self.parent = parent

    def sample_action(self, state):
        return np.argmax(self.parent.get_action_probs(state))

    def get_action_probs(self, state):
        return [1 if i == self.sample_action(state) else 0 for i in range(self.parent.num_actions)]

    def get_action_prob(self, state, action):
        return 1 if action == self.sample_action(state) else 0

    def set_action(self, state, action):
        if self.parent.get_action_prob(state, action) == 1:
            return False

        for a in range(self.parent.num_actions):
            self.parent.policy[state][a] = 1 if a == action else 0
        return True

class EpsilonGreedy(object):

    def __init__(self, parent, epsilon=0.5):
        self.parent = parent
        self.epsilon = epsilon

    def sample_action(self, state):
        return np.random.choice(range(self.parent.num_actions), p=self.get_action_probs(state))

    def get_action_probs(self, state):
        optimal_action = np.argmax(self.parent.get_action_probs(state))
        probs = [(1-self.epsilon) + (self.epsilon/self.parent.num_actions) if a == optimal_action else \
                (self.epsilon/self.parent.num_actions) for a in range(self.parent.num_actions)]
        return probs

    def get_action_prob(self, state, action):
        return self.get_action_probs(state)[action]

    def set_action(self, state, action):
        if np.isclose(1-self.epsilon + (self.epsilon / self.parent.num_actions),
                      self.parent.get_action_prob(state, action)):
            return False

        for a in range(self.parent.num_actions):
            if a == action:
                self.parent.policy[state][a] = 1 - self.epsilon + (self.epsilon / self.parent.num_actions)
            else:
                self.parent.policy[state][a] = self.epsilon / self.parent.num_actions
        return True

class TabularPolicy(Policy):

    def __init__(self, state_space, action_space):
        if not isinstance(state_space, discrete.Discrete):
            raise TypeError("the state space must be of type {}".format(discrete.Discrete))
        if not isinstance(action_space, discrete.Discrete):
            raise TypeError("the action space must be of type {}".format(discrete.Discrete))

        super(TabularPolicy, self).__init__(state_space, action_space)

        self.num_states = state_space.n
        self.num_actions = action_space.n
        self.policy = [ [(1/action_space.n) for _ in range(action_space.n)] for _ in range(state_space.n) ]

    def sample_action(self, state):
        return np.random.choice(range(len(self.policy[state])) , p=self.policy[state])

    def get_action_probs(self, state):
        return self.policy[state]

    def get_action_prob(self, state, action):
        return self.policy[state][action]

    def get_policy(self):
        return self.policy

    def set_policy(self, policy):
        self.policy = policy


class GridPolicy(TabularPolicy):

    def __init__(self, state_space, action_space, shape):
        if not (shape[0] * shape[1]) == state_space.n:
            raise ValueError("nRows x nCols given by shape does not much the size of the state space")

        self.shape = shape

        super(GridPolicy, self).__init__(state_space, action_space)

    def display(self):
        action_symbols = ['^', 'v', '<', '>']
        for state in range(self.num_states):
            if (not state == 0) and (state % self.shape[1] == 0):
                print()
            best_action = self.greedy.sample_action(state)
            print(action_symbols[best_action], end = "")
        print()
