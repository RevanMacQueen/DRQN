import random
import numpy as np
import torch
from collections import namedtuple, deque
from agent.settings import device


class ReplayBuffer:
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''
        Initialize a ReplayBuffer object.
        
        parameters:
            action_size : (int) dimension of each action
            buffer_size : (int) maximum size of buffer
            batch_size : (int) size of each training batch
            seed : (int) random seed
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''
        Add a new experience to memory
        
        parameters:
            state : (np.Array) the state observation
            action : (int) action chosen at obs
            reward : (float) the reward upon taking aciton 
            next_state : (np.Array) the next observation after obs
            done : (bool) whether the episode is done
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''
        Randomly sample a batch of experiences from memory

        returns:
            (states, actions, rewards, next_states, dones) : (tuple) each element of the tuple is a list
        '''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def can_sample(self):
        '''
        Determines if a valid batch can be produced from the current buffer

        returns: (bool)
        '''
        return len(self.memory) > self.batch_size

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)


class RNNReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seq_len, seed):
        '''
        Initialize a RNNReplayBuffer object. The buffer_size determines the number of episodes of 
        experience whic are stored. Each episode may have an arbitrary length. 
      
        parameters:
            action_size : (int) dimension of each action
            buffer_size : (int) maximum size of buffer
            batch_size : (int) size of each training batch
            seq_len : (int) the length of the trajectories used for training
            seed : (int) random seed
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.memory.append([])
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add_episode(self, episode=None):
        '''
        Add a new episode to the replay buffer. The oldest episode is evicted.

        parameters:
            episode : (list) a list of transition tuples
        '''
        if episode:
            self.memory.append(episode)
        else:
            self.memory.append([])

    def add(self, state, action, reward, next_state, done):
        '''
        Add a new piece of experience to the replay buffer. The experience is added to the most recent episode

        parameters:
            state : (np.Array) the state observation
            action : (int) action chosen at obs
            reward : (float) the reward upon taking aciton 
            next_state : (np.Array) the next observation after obs
            done : (bool) whether the episode is done
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory[-1].append(e)
        if done:
            self.memory.append([])

    def can_sample(self):
        '''
        Determines if a valid batch can be produced from the current buffer. 

        returns: (bool)
        '''
        for episode in self.memory:
            if len(episode) >= self.seq_len:
                return True
        return False

    def sample(self):
        '''
        Generate a sample of batch_size elements. Samples episodes with replacement.
        
        returns:
            tuple of the form: (state_sample, action_sample, reward_sample, next_state_sample, done_sample)
        '''
        valid_episodes = [episode for episode in self.memory if len(
            episode) >= self.seq_len]
        episodes = random.choices(valid_episodes, k=self.batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for episode in episodes:
            seq_start = random.randint(0, len(episode)-self.seq_len)
            experiences = episode[seq_start:seq_start+self.seq_len]

            state_batch.append(
                np.vstack([e.state for e in experiences if e is not None]))
            action_batch.append(
                np.vstack([e.action for e in experiences if e is not None]))
            reward_batch.append(
                np.vstack([e.reward for e in experiences if e is not None]))
            next_state_batch.append(
                np.vstack([e.next_state for e in experiences if e is not None]))
            done_batch.append(
                np.vstack([e.done for e in experiences if e is not None]))

        state_batch = torch.from_numpy(
            np.stack(state_batch)).float().to(device)
        action_batch = torch.from_numpy(
            np.stack(action_batch)).long().to(device)
        reward_batch = torch.from_numpy(
            np.stack(reward_batch)).float().to(device)
        next_state_batch = torch.from_numpy(
            np.stack(next_state_batch)).float().to(device)
        done_batch = torch.from_numpy(np.stack(done_batch)).float().to(device)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        '''
        Returns the total number of experiences in the buffer across all episodes.
        '''
        return sum([len(episode) for episode in self.memory])


class FixedLengthRNNReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seq_len, seed):
        '''
        Initialize a FixedLengthRNNReplayBuffer. The buffer_size determines the number of episodes of 
        experience whic are stored. Each episode may have an arbitrary length but the total number of 
        transitions is fixed
        
        parameters:
            action_size : (int) dimension of each action
            buffer_size : (int) maximum size of buffer (transitions)
            batch_size : (int) size of each training batch
            seq_len : (int) the length of the trajectories used for training
            seed : (int) random seed
        '''
        self.action_size = action_size
        self.memory = [[]]
        self.sizes = [0]  # size of each episode in the RB

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add_episode(self, episode=None):
        '''
        Add a new episode to the replay buffer. The oldest episode is evicted.

        parameters:
            episode : (list) a list of transition tuples
        '''
        if episode:
            self.memory.append(episode)
            self.sizes.append(len(episode))

            # if over capacity, remove elements until under capacity
            while sum(self.sizes) >= self.buffer_size:
                self.memory.pop(0)
                self.sizes.pop(0)
        else:
            self.memory.append([])
            self.sizes.append(0)

    def add(self, state, action, reward, next_state, done):
        '''
        Add a new piece of experience to the replay buffer. The experience is added to the most recent episode

        parameters:
            state : (np.Array) the state observation
            action : (int) action chosen at obs
            reward : (float) the reward upon taking aciton 
            next_state : (np.Array) the next observation after obs
            done : (bool) whether the episode is done
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory[-1].append(e)
        self.sizes[-1] += 1

        if done:  # new episode
            self.end_episode()

        # if over capacity, remove elements until under capacity
        while sum(self.sizes) >= self.buffer_size:
            self.memory.pop(0)
            self.sizes.pop(0)

    def end_episode(self):
        '''
        Adds a new episode to the buffer
        '''
        self.memory.append([])
        self.sizes.append(0)

    def can_sample(self):
        '''
        Determines if a valid batch can be produced from the current buffer. 

        returns: (bool)
        '''
        for episode in self.memory:
            if len(episode) >= self.seq_len:
                return True
        return False

    def sample(self):
        '''
        Generate a sample of batch_size elements. Samples episodes with replacement.
        
        returns:
            tuple of the form: (state_sample, action_sample, reward_sample, next_state_sample, done_sample)
        '''
        valid_episodes = [episode for episode in self.memory if len(
            episode) >= self.seq_len]
        episodes = random.choices(valid_episodes, k=self.batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for episode in episodes:
            seq_start = random.randint(0, len(episode)-self.seq_len)
            experiences = episode[seq_start:seq_start+self.seq_len]

            state_batch.append(
                np.vstack([e.state for e in experiences if e is not None]))
            action_batch.append(
                np.vstack([e.action for e in experiences if e is not None]))
            reward_batch.append(
                np.vstack([e.reward for e in experiences if e is not None]))
            next_state_batch.append(
                np.vstack([e.next_state for e in experiences if e is not None]))
            done_batch.append(
                np.vstack([e.done for e in experiences if e is not None]))

        state_batch = torch.from_numpy(
            np.stack(state_batch)).float().to(device)
        action_batch = torch.from_numpy(
            np.stack(action_batch)).long().to(device)
        reward_batch = torch.from_numpy(
            np.stack(reward_batch)).float().to(device)
        next_state_batch = torch.from_numpy(
            np.stack(next_state_batch)).float().to(device)
        done_batch = torch.from_numpy(np.stack(done_batch)).float().to(device)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        '''
        Returns the total number of experiences in the buffer across all episodes.
        '''
        return sum(self.sizes)
