#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:21:30 2021

@author: kerrick
"""
import random
from collections import namedtuple, deque

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class RNNReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seq_len, seed):
        """Initialize a RNNReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seq_len (int): the length of the trajectories used for training
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.trajectory = namedtuple("Trajectory", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        t = self.trajectory(states, actions, rewards, next_states, dones)
        self.memory.append(t)

    def sample(self):
        trajectories = random.sample(self.memory, k=self.batch_size)

        state_sequences = torch.from_numpy(np.vstack([t.states for t in trajectories])).float().to(device)
        action_sequences = torch.from_numpy(np.vstack([t.actions for t in trajectories])).float().to(device)
        reward_sequences = torch.from_numpy(np.vstack([t.rewards for t in trajectories])).float().to(device)
        next_state_sequences = torch.from_numpy(np.vstack([t.next_states for t in trajectories])).float().to(device)
        done_sequences = torch.from_numpy(np.vstack([t.dones for t in trajectories])).float().to(device)

        return (state_sequences, action_sequences, reward_sequences, next_state_sequences, done_sequences)

    def __len__(self):
        return len(self.memory)