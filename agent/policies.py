import numpy as np
from scipy.special import softmax


def random_argmax(a):
    '''
    like np.argmax, but returns a random index in the case of ties

    parameters:
        a : (np.Array) 

    '''
    return np.random.choice(np.flatnonzero(a == a.max()))


class EGreedyPolicy(object):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def get_p(self, q):
        '''
        Return probability distribution p over actions representing a stochastic policy

        parameters:
            q : (np.Array) values for each action for a fixed state

        returns:
            p : (np.Array) probability of each action
        '''

        q = q.flatten()
        num_actions = q.shape[0]
        p = np.ones(q.shape) * (self.epsilon/num_actions)

        argmax_a = random_argmax(q)
        p[argmax_a] += (1 - self.epsilon)

        return p

    def get_action(self, q):
        '''
        Select action accoring to policy

        parameters:
            q : (np.Array) values for each action for a fixed state

        returns:
            a : (int) the selected action
        '''

        p = self.get_p(q)
        num_actions = q.shape[0]
        a = np.random.choice(num_actions, p=p)
        return a


class SoftmaxPolicy(object):
    def __init__(self, eta=1):
        self.eta = eta

    def get_p(self, q):
        '''
        Return probability distribution p over actions representing a stochastic policy

        parameters:
            q : (np.Array) values for each action for a fixed state

        returns:
            p : (np.Array) probability of each action
        '''

        q = q.flatten()
        p = softmax(self.eta*q)  # probability of choosing each action

        return p

    def get_action(self, q):
        '''
        Select action accoring to policy

        parameters:
            q : (np.Array) values for each action for a fixed state

        returns:
            a : (int) the selected action
        '''

        p = self.get_p(q)
        num_actions = q.shape[0]
        a = np.random.choice(num_actions, p=p)
        return a
