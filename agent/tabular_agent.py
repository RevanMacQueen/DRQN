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
        self.step_size = self.agent_params['step_size']

        self.policy = EGreedyPolicy(epsilon=self.eps)
        self.q = np.zeros([self.input_dim, self.action_dim])

    def act(self, obs):
        """
        Take an returns an action given observation obs. 
        Assume obs is an integer value
        """
        a = self.policy.get_action(self.q[obs,:])
        return a
        
    def act_evaluate(self, obs):
        """
        Like act(), but acts greedily
        """
        return random_argmax(self.q[obs,:])

    def train_step(self, obs, action, reward, next_obs, done):
        """
        Updates q values
        """

        self.q[obs, action] +=  self.step_size*(reward + self.gamma*np.max(self.q[next_obs, :]) - self.q[obs, action]) # Q-learning update
