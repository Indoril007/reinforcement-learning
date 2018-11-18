import numpy as np
import tkinter
from tkinter.font import Font
from .tabular import TabularAgent

class GridAgent(TabularAgent):

    def __init__(self, shape: tuple, discount: float) -> None:
        """

        :param shape: The shape of the grid world given as (number of rows, number of columns)
        :param discount: The disocunt factor to be used in calculating return
        """
        self.nrows, self.ncols = shape
        super(GridAgent, self).__init__(self.ncols*self.nrows, 4, discount)

    def ij2state(self, i: int, j: int) -> int:
        """
        returns the state value given the row and column number
        :param i: row number
        :param j: column number
        :return: state value
        """
        return i*self.ncols+j

    def state2ij(self, state: int) -> tuple:
        """
        Converts the state value to the row and column number
        :param state: The state value
        :return: The row and column number
        """
        i = state // self.ncols
        j = state % self.ncols
        return i, j

    def display_values(self, window: tkinter.Tk, true_values: list = None) -> None:
        """
        This function displays values on top of the rendered grid world returned from env.render
        :param window: This is the window object (tkinter.Tk) which is returns from an env.render call to a griddworld
            environment
        :param true_values: The true values (likely calculated by Dynamic methods) for which to calculate the error with
        """
        if not (self.nrows == window.nrows and self.ncols == window.ncols):
            raise ValueError("Dimensions of the window grid do not match up with the agent's grid")
        for i in range(self.nrows):
            for j in range(self.ncols):
                state = self.ij2state(i,j)
                x = j*window.unit + 3
                y = (i+1)*window.unit

                if true_values is not None:
                    error = np.abs(true_values[state] - self.values[state])
                    color = error2color(error)
                else:
                    color = "#000000"
                text = "{:.3f}".format(self.values[state])
                window.canvas.values_text = window.canvas.create_text(x,y, anchor=tkinter.SW, fill=color, text=text)
                window.update_idletasks()
                window.update()

    def display_q_values(self, window = None, true_values = None):
        pass

    def display_policy(self, window = None, optimal_policy=None):
        if not (self.nrows == window.nrows and self.ncols == window.ncols):
            raise ValueError("Dimensions of the window grid do not match up with the agent's grid")

        arrows = ["^", "v", "<", ">"]
        pi = self.policy.get()
        for i in range(self.nrows):
            for j in range(self.ncols):
                state = self.ij2state(i,j)
                x = (j + 0.5) * window.unit
                y = (i + 0.5) * window.unit

                action = np.argmax(pi[state])
                text = arrows[action]
                font = Font(size=20)
                window.canvas.policy_text = window.canvas.create_text(x, y, font=font, anchor=tkinter.CENTER, text=text)
                window.update_idletasks()
                window.update()









def error2color(self, error):
    error = min(1, error)
    red = "{:0>2x}".format(int(error*255))
    green = "{:0>2x}".format(int((1-error)*255))
    return "#" + red + green + "00"




