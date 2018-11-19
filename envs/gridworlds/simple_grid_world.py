import numpy as np
import tkinter as tk
import gym
from gym.envs.toy_text import discrete

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

UNIT = 150

DEFAULT_END_STATES = [5]

DEFAULT_WORLD = np.array([
    [0, 0, 0, 0, 0, 'd'],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

WORLD_2 = np.array([
    [0, 0, 0, 0, 0, 'd'],
    [0, 0, 0, 0, 0, 0],
    [0, 0, -1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

WORLD_1 = np.array([
    [0, 0, 0, 0, 0, 'd'],
    [0, 0, -3, 0, -1, -1],
    [0, 0, -3, 0, 0, 0],
    [0, 0, -3, -3, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

RANDOM_START_ISD = np.full((6,6), (1/35))
RANDOM_START_ISD[0,5] = 0

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

    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(self, world_array=WORLD_1, isd=DEFAULT_ISD, action_error = 0.0):
        assert type(world_array) is np.ndarray
        assert type(isd) is np.ndarray
        self.end_states = DEFAULT_END_STATES
        self.nrows, self.ncols = world_array.shape
        self.shape = world_array.shape
        self.world_array = world_array
        self.nS = world_array.size
        self.nA = 4
        self.isd = isd
        self.action_error = action_error
        self.P = [ [ [] for _ in range(self.nA)  ] for _ in range(self.nS) ]
        self.master = None

        self._init_transitions()
        self.numpized_transitions = np.array(self.P)

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
                    transitions.append([transition_prob, new_state, reward, done])

    def _to_state(self, row, col):
        return row*self.ncols + col

    def _to_coords(self, state):
        i = state // self.ncols
        j = state % self.ncols
        return (i,j)

    def _move(self, row, col, action):
        if action == UP:
            row = max(row-1,0)
        elif action == DOWN:
            row = min(row+1, self.nrows-1)
        elif action == LEFT:
            col = max(col-1, 0)
        elif action == RIGHT:
            col = min(col+1, self.ncols-1)
        return (row,col)

    def render(self, mode='ansi', window = None):
        if mode == 'ansi':
            string = ""
            for i, row in enumerate(self.world_array):
                for j, el in enumerate(row):
                    string += "X" if self._to_state(i,j) == self.s else str(self.world_array[i, j])
                string += "\n"
            return string
        elif mode == 'human':
            if window.canvas is None:
                window.canvas = tk.Canvas(window, bg='white', height=self.nrows*UNIT, width=self.ncols*UNIT)

                for col in range(0, self.ncols*UNIT, UNIT):
                    x0, y0, x1, y1 = col, 0, col, self.nrows*UNIT
                    window.canvas.create_line(x0, y0, x1, y1)
                for row in range(0, self.nrows*UNIT, UNIT):
                    x0, y0, x1, y1 = 0, row, self.ncols*UNIT, row
                    window.canvas.create_line(x0, y0, x1, y1)
            else:
                window.canvas.delete(window.canvas.state_drawing)

            i, j = self._to_coords(self.s)
            center_y = (i+0.5)*UNIT
            center_x = (j+0.5)*UNIT
            x0 = center_x - UNIT*0.1
            x1 = center_x + UNIT*0.1
            y0 = center_y - UNIT*0.1
            y1 = center_y + UNIT*0.1
            window.canvas.state_drawing = window.canvas.create_oval([x0, y0, x1, y1])
            window.unit = UNIT
            window.ncols = self.ncols
            window.nrows = self.nrows

            window.canvas.pack()
            window.update_idletasks()
            window.update()



