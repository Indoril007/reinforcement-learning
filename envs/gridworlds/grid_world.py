import gym
from gym.envs.toy_text import discreet

class GridWorld():

    actions = ['Left', 'Down', 'Right', 'Up']

    def __init__(self, world):

    self.world = world
    self.shape = world.shape


    nA = 4

    super(GridWorld, self).__init__(nS, nA, P, isd)
