from gym import Env, spaces
import numpy as np
import matplotlib.pyplot as plt
import random

from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms
from mazelib.transmute.Perturbation import Perturbation
from mazelib.generate.Prims import Prims

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3  # agents actions


class RandomMaze(Env):
    '''
    A randomly generated maze gridworld environment using OpenAI gym API
    '''

    def __init__(self, n=10, cycles=3, seed=np.random.randint(0, 10000), state_representation='integer'):
        '''
        Initialize a maze 

        parameters : 
            n : (int) Maze has diameter 2n+1
            cycles : (int) number of blocks to remove to create cycles
            seed : (int) seed for random initialization
            state_representation : (str) how state ought to be represented for the agent
        '''

        Maze.set_seed(seed)
        m = Maze()
        m.generator = DungeonRooms(n, n)
        m.generate()
        self.grid = m.grid

        # remove some walls to create cycles
        walls = np.argwhere(self.grid[1:(2*n), 1:(2*n)])
        np.random.shuffle(walls)
        for i in range(cycles):
            row, col = walls[i]
            row += 1
            col += 1
            self.grid[row, col] = 0

        corners = [[1, 1], [1, 2*n-1], [2*n-1, 1], [2*n-1, 2*n-1]]

        self.start = random.choice(corners)
        corners.remove(self.start)
        self.end = random.choice(corners)

        self.loc = self.start  # location of agent

        self.action_space = spaces.Discrete(4)
        if state_representation == 'integer':
            self.gen_state = self.gen_integer_state
            self.observation_space = spaces.Discrete(np.prod(self.grid.shape))
        elif state_representation == 'one_hot':
            self.gen_state = self.gen_one_hot_state
            self.observation_space = spaces.Discrete(np.prod(self.grid.shape))
            self.observation_space = spaces.Box(low=0, high=1, shape=(
                np.prod(self.grid.shape), ), dtype=np.int8)
        elif state_representation == 'flat_grid':
            self.gen_state = self.gen_flat_grid_state
            self.observation_space = spaces.Box(low=0, high=5, shape=(
                np.prod(self.grid.shape), ), dtype=np.int8)  # not sure if this is right?
        else:
            raise NotImplementedError  # add other ways to represent state here

        self.episode_length = 0
        self.MAX_LENGTH = 1000 # episodes are cut off at this length

    def reset(self):
        '''
        Resets environment before an episode
        '''
        self.loc = self.start
        self.episode_length = 0
        return self.gen_state(self.loc)

    def step(self, action):
        '''
        updates state given an action by the agent

        parameters:
            action : (int)
        '''
        self.episode_length += 1
        row, col = self.loc  # row major format

        if action == UP:
            row -= 1
        elif action == RIGHT:
            col += 1
        elif action == DOWN:
            row += 1
        elif action == LEFT:
            col -= 1

        if self.grid[row, col]:  # if hit a wall
            row, col = self.loc

        if [row, col] == self.end:
            reward = 1
            is_done = True
        else:
            reward = 0
            is_done = False

        self.loc = [row, col]

        if self.episode_length >= self.MAX_LENGTH:
            is_done = True

        return self.gen_state(self.loc), reward, is_done, None

    def gen_integer_state(self, loc):
        '''
        Returns an integer for the current state

        NOTE: returns a numpy array with one element. This is done so that  NN-based methods using pytorch 
        can also use this representation
        
        parameters:
            loc : (list) [row, column] location 
        '''
        row, col = loc

        return np.array([row * self.grid.shape[1] + col])

    def num_states(self):
        '''
        returns the number of tiles in the maze. NOTE: this include walls
        '''
        return np.prod(self.grid.shape)

    def gen_one_hot_state(self, loc):
        '''
        Generates a one-hot encoding of state

        parameters:
            loc : (list) [row, column] location 
        '''
        row, col = loc
        ind = [row * self.grid.shape[1] + col]
        one_hot = np.zeros(self.num_states())
        one_hot[ind] = 1
        return one_hot

    def gen_flat_grid_state(self, loc):
        '''
        returns a flattened representation of the entire maze

        parameters:
            loc : (list) [row, column] location 
        '''

        grid = np.copy(self.grid)

        plt.figure(figsize=(10, 5))

        grid[self.start[0], self.start[1]] = 2
        grid[self.end[0], self.end[1]] = 3
        grid[self.loc[0], self.loc[1]] = 4

        return grid.flatten()

    def showPNG(self):
        '''
        Generate a simple image of the maze
        '''

        grid = np.copy(self.grid)

        plt.figure(figsize=(10, 5))

        grid[self.start[0], self.start[1]] = 2
        grid[self.end[0], self.end[1]] = 3
        grid[self.loc[0], self.loc[1]] = 4

        plt.imshow(grid, interpolation='nearest')

        plt.xticks([]), plt.yticks([])
        plt.show()