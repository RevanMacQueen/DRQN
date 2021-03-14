#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:20:34 2021

@author: kerrick
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=64):
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
        
        self.fc1 = nn.Linear(self.state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_values = self.fc3(x)
        
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
        self.rnn = nn.RNN(self.input_size, self.hidden_state_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_state_size, self.action_size)

        self.hidden = self.init_hidden(1) # hidden state for prediction, not learning


    def forward(self, x, h0):
        """
        Forward pass for training
        """
        # in prediction, is this actually using the hidden state?
        # TODO might need to use https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence 

        
        # if len(x.shape) < 3:
        #     x = x.unsqueeze(0)
        # hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x,h0)
        #self.hidden = hidden
        #out = out.view(-1, self.hidden_state_size) # idk what this line does 
        unpacked_out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=0.0)
        action_values = self.fc(unpacked_out[0])
        return action_values, unpacked_out[1]

    def forward_prediction(self, x):
         
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        out, self.hidden = self.rnn(x, self.hidden)
        #out = out.view(-1, self.hidden_state_size)
        
        action_values = self.fc(out)
        return action_values


    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_state_size).to(device)
        return hidden