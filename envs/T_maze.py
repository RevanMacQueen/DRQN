from gym import Env, spaces
import numpy as np
import matplotlib.pyplot as plt



class TMaze(Env):
    """
    A randomly generated maze gridworld environment
    """

    def __init__(self, n=6):

        """
        T-maze environment from Reinforcement Learning with Long Short-Term Memory 
        by Bram Bakker
        
        n is the length of the corridor
        """
        
        self.corridor_len = n
        self.start = 0 
        self.loc = self.start # location of agent
        self.goal_pos = np.random.choice([0,1])

        self.UP, self.RIGHT, self.DOWN, self.LEFT = 0, 1, 2, 3 # agents actions

        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(low=0, high=1, shape = (1,3), dtype=np.int8)
        
        
    def reset(self):
        self.loc = self.start
        self.goal_pos = np.random.choice([0,1])
        return  self.gen_state(self.loc)
    
    def step(self, action):
        
        reward = -0.1   # default
        is_done = False # default
        if action == self.UP:
            if self.loc == self.corridor_len:
                self.loc = -1
                if self.goal_pos == 0:
                    # success
                    reward = 1
                    is_done = True
                else:
                    reward = -1
                    is_done = True
                    
        elif action == self.RIGHT:
            if self.loc < self.corridor_len:
                self.loc += 1
                
        elif action == self.DOWN:
            if self.loc == self.corridor_len:
                self.loc = -2
                if self.goal_pos == 1:
                    # success
                    reward = 1
                    is_done = True
                else:
                    reward = -1
                    is_done = True
            
        elif action == self.LEFT:
            if self.loc > 0:
                self.loc -= 1

        return self.gen_state(self.loc), reward, is_done, None


    def gen_state(self, loc):
        if loc == 0:
            if self.goal_pos == 0:
                state = np.array([0,1,1])
            else:
                state = np.array([1,1,0])
                
        elif loc < self.corridor_len:
            state = np.array([1,0,1])
            
        elif loc == self.corridor_len:
            state = np.array([0,1,0])
        if loc < 0:
            state = np.array([0,0,0])  # terminal
        
        return state
    

    def num_states(self):
        return self.corridor_len+1


    def showPNG(self):
        """Generate a simple image of the maze."""

        N = self.corridor_len+3
        grid = np.zeros((N,N)) 
        grid[N//2,1:self.corridor_len+2] = -1
        if self.goal_pos == 0:
            grid[N//2-1,self.corridor_len+1] = 2
            grid[N//2+1,self.corridor_len+1] = -1
        else:
            grid[N//2-1,self.corridor_len+1] = -1
            grid[N//2+1,self.corridor_len+1] = 2
        
        #grid[N//2-1:N//2+2,self.corridor_len+1] = 1
        
        # place agent
        if self.loc == -1:
            grid[N//2-1,1+self.corridor_len] = 3
        elif self.loc == -2:
            grid[N//2+1,1+self.corridor_len] = 3
        else:
            grid[N//2,1+self.loc] = 3
        
        plt.figure(figsize=(10, 5))

        plt.imshow(grid, interpolation='nearest')

        plt.xticks([]), plt.yticks([])
        plt.show()