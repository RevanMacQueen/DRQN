from gym import Env, spaces
import numpy as np
import matplotlib.pyplot as plt
import random

from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms
from mazelib.transmute.Perturbation import Perturbation
from mazelib.generate.Prims import Prims


class RandomMaze(Env):
    """
    A randomly generated maze gridworld environment
    """

    def __init__(self, n=10, cycles=3, seed=np.random.randint(0, 10000)):

        """
        Use prims algorithm to generate the maze.

        Maze has diameter 2n+1
        cycles is number of blocks to remove to create cycles
        """

        Maze.set_seed(seed)
        m = Maze()
        m.generator = DungeonRooms(n, n)
        m.generate()
        
        self.grid = m.grid

        # remove some walls to create cycles
        walls = np.argwhere(self.grid[1:(2*n), 1:(2*n)])
        np.random.shuffle(walls)
        for i in range(cycles):
            row, col  = walls[i]
            row += 1
            col += 1
            self.grid[row, col] = 0

        corners = [[1,1], [1,2*n-1], [2*n-1,1], [2*n-1,2*n-1]]

        self.start = random.choice(corners)
        corners.remove(self.start)
        self.end = random.choice(corners)

        self.loc = self.start # location of agent

        self.UP, self.RIGHT, self.DOWN, self.LEFT = 0, 1, 2, 3 # agents actions


        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.prod(self.grid.shape))

    def reset(self):
        self.loc = self.start
        return  np.array(self.loc)
    
    def step(self, action):
        row,col = self.loc # row major format
        
        if action == self.UP:
            row -= 1
        elif action == self.RIGHT:
            col += 1
        elif action == self.DOWN:
            row += 1
        elif action == self.LEFT:
            col -= 1

        if self.grid[row, col]: # if hit a wall
            row, col = self.loc

        if [row, col] == self.end:
            reward = 1
            is_done = True
        else:
            reward = 0
            is_done = False

        self.loc = [row, col]

        return np.array(self.loc), reward, is_done, None

    def showPNG(self):
        """Generate a simple image of the maze."""

        grid = np.copy(self.grid) 

        plt.figure(figsize=(10, 5))

        grid[self.start[0], self.start[1]] = 2
        grid[self.end[0], self.end[1]] = 3
        grid[self.loc[0], self.loc[1]] = 4

        plt.imshow(grid, interpolation='nearest')

        plt.xticks([]), plt.yticks([])
        plt.show()

