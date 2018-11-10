import numpy as np
import gym
from gym.envs.toy_text import discrete

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DEFAULT_END_STATES = [5]

DEFAULT_WORLD = np.array([
    [0, 0, 0, 0, 0, 'd'],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

DEFAULT_ISD = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
]).ravel()

class SimpleGridWorld(discrete.DiscreteEnv):

    nA = 4 # Four actions: up, down, left, right

    metadata = {'render.modes': ['ansi']}

    def __init__(self, world_array=DEFAULT_WORLD, isd=DEFAULT_ISD, action_error = 0.0):
        assert type(world_array) is np.ndarray
        assert type(isd) is np.ndarray
        self.end_states = DEFAULT_END_STATES
        self.nrow, self.ncol = world_array.shape
        self.shape = world_array.shape
        self.world_array = world_array
        self.nS = world_array.size
        self.nA = 4
        self.isd = isd
        self.action_error = action_error
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        self._init_transitions()

        super(SimpleGridWorld, self).__init__(self.nS, self.nA, self.P, self.isd)

    def _init_transitions(self):
        for (i, j), el in np.ndenumerate(self.world_array):
            state = self._to_state(i,j)

            for action in range(4):
                transitions = self.P[state][action]

                for direction in range(4):
                    new_i, new_j = self._move(i,j,direction)
                    new_state = self._to_state(new_i, new_j)
                    state_val = self.world_array[new_i, new_j]
                    reward = int(state_val) if state_val != 'd' else 1
                    done = state_val == "d"
                    transition_prob = 1-self.action_error if action==direction else self.action_error / 3
                    transitions.append((transition_prob,  new_state, reward, done))

    def _to_state(self, row, col):
        return row*self.ncol + col

    def _move(self, row, col, action):
        if action == UP:
            row = max(row-1,0)
        elif action == DOWN:
            row = min(row+1, self.nrow-1)
        elif action == LEFT:
            col = max(col-1, 0)
        elif action == RIGHT:
            col = min(col+1, self.ncol-1)
        return (row,col)

    def render(self, mode='ansi'):
        string = ""
        for i, row in enumerate(self.world_array):
            for j, el in enumerate(row):
                string += "X" if self._to_state(i,j) == self.s else str(self.world_array[i, j])
            string += "\n"
        return string

