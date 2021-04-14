#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:20:34 2021

@author: kerrick
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.settings import device

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, num_layers=1, hidden_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        self.input_layer = nn.Linear(self.state_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(num_layers-1)]) # (additional) hidden layers
        self.final = nn.Linear(self.hidden_size, self.action_size) #final layer

        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.input_layer(state)
        x = F.relu(x)

        for i,l in enumerate(self.hidden_layers):
            x = l(x)
            x = F.relu(x)
        action_values = self.final(x)
        
        return action_values

class RNNQNetwork(nn.Module):
    """Simple recurrent network with single RNN layer and single linear layer"""

    def __init__(self, input_size, action_size, hidden_state_size, seed, num_layers=1):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): Dimension of each state x number of states in sequence
            action_size (int): Dimension of each action, also the size of the network output
            hidden_state_size (int): Dimension of the RNN hidden state
            seed (int): Random seed
            num_layers (int): The number of recurrent layers (currently unused)
        """
        super(RNNQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_size = input_size
        self.action_size = action_size
        # self.hidden_layer_size = hidden_layer_size
        self.hidden_state_size = hidden_state_size
        self.num_layers = num_layers
        
        self.initial = nn.Linear(self.input_size, self.hidden_state_size) #initial layer
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_state_size, self.hidden_state_size) for i in range(num_layers-1)]) # additional hidden layers

        self.rnn = nn.RNN(self.hidden_state_size, self.hidden_state_size, batch_first=True, nonlinearity='relu')

        self.final = nn.Linear(self.hidden_state_size, self.action_size) #final layer

        self.hidden = self.init_hidden(1) # hidden state for prediction, not learning


    def forward(self, x):
        """
        Forward pass for training
        """
        # in prediction, is this actually using the hidden state?
        # TODO might need to use https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence 

        
        if len(x.shape) < 3:
            x = x.unsqueeze(0)

        x = self.initial(x)
        x = F.relu(x)
        for i, l in enumerate(self.hidden_layers):
            x = l(x)
            x = F.relu(x)

        out, hidden = self.rnn(x)
        action_values = self.final(out)

        return action_values

    def forward_prediction(self, x):
         
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        x = self.initial(x)
        x = F.relu(x)
        for i, l in enumerate(self.hidden_layers):
            x = l(x)
            x = F.relu(x)

        out, self.hidden = self.rnn(x, self.hidden)
    
        action_values = self.final(out)
        return action_values


    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_state_size).to(device)
        return hidden


class RNNQNetworkZeroState(nn.Module):
    """Simple recurrent network with single RNN layer and single linear layer"""

    def __init__(self, input_size, action_size, hidden_state_size, seed, num_layers=1):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): Dimension of each state x number of states in sequence
            action_size (int): Dimension of each action, also the size of the network output
            hidden_state_size (int): Dimension of the RNN hidden state
            seed (int): Random seed
            num_layers (int): The number of recurrent layers (currently unused)
        """
        super(RNNQNetworkZeroState, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_size = input_size
        self.action_size = action_size
        # self.hidden_layer_size = hidden_layer_size
        self.hidden_state_size = hidden_state_size
        self.num_layers = num_layers
        
        self.initial = nn.Linear(self.input_size, self.hidden_state_size) #initial layer
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_state_size, self.hidden_state_size) for i in range(num_layers-1)]) # additional hidden layers

        self.rnn = nn.RNN(self.hidden_state_size, self.hidden_state_size, batch_first=True, nonlinearity='relu')

        self.final = nn.Linear(self.hidden_state_size, self.action_size) #final layer

        self.hidden = self.init_hidden(1) # hidden state for prediction, not learning


    def forward(self, x):
        """
        Forward pass for training
        """
        # in prediction, is this actually using the hidden state?
        # TODO might need to use https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence 

        
        if len(x.shape) < 3:
            x = x.unsqueeze(0)

        x = self.initial(x)
        x = F.relu(x)
        for i, l in enumerate(self.hidden_layers):
            x = l(x)
            x = F.relu(x)

        out = torch.zeros(x.shape).to(device)
        for i in range(x.shape[1]):       
            oneStep_in = x[:,i,:].unsqueeze(1)
            oneStep_hidden = self.init_hidden(x.shape[0])
            oneStep_out, hidden = self.rnn(oneStep_in,oneStep_hidden)
            out[:,i,:] = oneStep_out.squeeze(1)
        
        action_values = self.final(out)

        return action_values

    def forward_prediction(self, x):
         
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        x = self.initial(x)
        x = F.relu(x)
        for i, l in enumerate(self.hidden_layers):
            x = l(x)
            x = F.relu(x)

        out, self.hidden = self.rnn(x, self.init_hidden(1))
    
        action_values = self.final(out)
        return action_values


    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_state_size).to(device)
        return hidden