import random
import numpy as np

from agent.policies import EGreedyPolicy, random_argmax

class TabularAgent():  

    def __init__(self, agent_params):
        self.agent_params = agent_params

        self.input_dim = self.agent_params['input_dim']
        self.action_dim = self.agent_params['action_dim']
        self.eps = self.agent_params['epsilon']
        self.gamma = self.agent_params['gamma']
        self.step_size = self.agent_params['learning_rate']

        self.policy = EGreedyPolicy(epsilon=self.eps)
        self.q = np.zeros([self.input_dim, self.action_dim])

    def act(self, obs_arr):
        """
        Take an returns an action given observation obs. 
        params:
            obs_arr :np.array with a single entry giving the index of the state
        """
        obs = obs_arr[0]
        a = self.policy.get_action(self.q[obs,:])
        return a
        
    def act_evaluate(self, obs_arr):
        """
        Like act(), but acts greedily
        params:
            obs_arr :np.array with a single entry giving the index of the state
        """
        obs = obs_arr[0]
        return random_argmax(self.q[obs,:])

    def train_step(self, obs_arr, action, reward, next_obs_arr, done):
        """
        Updates q values
        params:
            obs_arr :np.array with a single entry giving the index of the state
        """
        obs = obs_arr[0]
        next_obs = next_obs_arr[0]
        self.q[obs, action] +=  self.step_size*(reward + self.gamma*np.max(self.q[next_obs, :]) - self.q[obs, action]) # Q-learning update
