import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.model import QNetwork, RNNQNetwork
from agent.replay_buffer import ReplayBuffer, RNNReplayBuffer
from agent.settings import device


class Agent():  
    def __init__(self, agent_params):
        self.agent_params = agent_params

        

        self.input_dim = self.agent_params['input_dim']
        self.action_dim = self.agent_params['action_dim']

        self.seed = self.agent_params.get('seed', np.random.randint(0,10000))
        if self.agent_params['model_arch'] == 'FFN':
            self.qnetwork_local = QNetwork(self.input_dim, self.action_dim, self.seed, num_layers=self.agent_params['num_layers']).to(device)
            self.qnetwork_target = QNetwork(self.input_dim, self.action_dim, self.seed, num_layers=self.agent_params['num_layers']).to(device)

            buffer_size = self.agent_params['buffer_size']
            batch_size = self.agent_params['batch_size']
           
            self.buffer = ReplayBuffer(self.action_dim, buffer_size, batch_size, self.seed)
            self.act = self.act_FFN

        elif self.agent_params['model_arch'] == 'RNN':
            
            self.hidden_layer_size = self.agent_params['hidden_layer_size'] 
            self.num_layers = self.agent_params['num_layers']
            buffer_size = self.agent_params['buffer_size']
            batch_size = self.agent_params['batch_size']
            seq_len = self.agent_params['seq_len']

            self.qnetwork_local  = RNNQNetwork(
                self.input_dim, 
                self.action_dim, 
                self.hidden_layer_size, 
                self.seed,
                num_layers=self.agent_params['num_layers']).to(device)

            self.qnetwork_target  = RNNQNetwork(
                self.input_dim, 
                self.action_dim, 
                self.hidden_layer_size, 
                self.seed,
                num_layers=self.agent_params['num_layers']).to(device)

            self.buffer = RNNReplayBuffer(self.action_dim, buffer_size, batch_size, seq_len, self.seed)
            self.prev_obs = np.zeros((seq_len, self.input_dim)) # a "buffer" of the previous number of sequences 
            self.act = self.act_RNN
            self.learn = self.learnRNN
        else:
            raise NotImplementedError
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.agent_params['learning_rate'])

        self.t_step = 0
        self.learning_starts = self.agent_params['learning_starts']
        self.learning_freq = self.agent_params['learning_freq']
        self.target_update_freq = self.agent_params['target_update_freq']
        self.eps = self.agent_params['epsilon']
        self.min_eps =  self.agent_params['min_epsilon']
        self.decay = self.agent_params['decay']
        self.gamma = self.agent_params['gamma']
        self.tau  = self.agent_params['tau']

    def act_FFN(self, obs):
        """
        Returns an action given observation obs using a FFN
        """
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(obs)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action =  random.choice(np.arange(self.action_dim))

        return action    

    def act_RNN(self, obs):
        """
        Returns an action given observation obs using a RNN. The difference with the above funtion is that 
        """
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward_prediction(obs)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action =  random.choice(np.arange(self.action_dim))

        return action    


    def train_step(self, obs, action, reward, next_obs, done, add_new_episode=False):
        """
        Handles agent training. Adds samples to replay buffer and, if appropriate, trains net
        """
        self.buffer.add(obs, action, reward, next_obs, done)

        if done: # if end of episode, decay epsilon by a factor of 0.99
            self.eps = self.eps * self.decay
            if self.eps < self.min_eps:
                self.eps = self.min_eps
            
            if self.agent_params['model_arch'] == 'RNN':
                self.qnetwork_local.eval()
                with torch.no_grad():
                    self.qnetwork_local.hidden = self.qnetwork_local.init_hidden(1) # if end of episode, refresh hidden state


        self.t_step += 1
        if self.t_step >= self.learning_starts and self.buffer.can_sample():
            if self.t_step % self.learning_freq == 0:
                experiences = self.buffer.sample()
                self.learn(experiences)


    def learnRNN(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # get targets
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_targets_next = torch.max(self.qnetwork_target.forward(next_states), dim=2, keepdim=True)[0]

        Q_targets = rewards + (self.gamma  * Q_targets_next * (1 - dones))

        # get outputs
        self.qnetwork_local.train()
        Q_expected = self.qnetwork_local.forward(states).gather(2, actions)

        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # clear gradients
        self.optimizer.zero_grad()

        # update weights local network
        loss.backward()

        # take one SGD step
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step % self.target_update_freq == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # get targets
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_targets_next = torch.max(self.qnetwork_target.forward(next_states), dim=1, keepdim=True)[0]

        Q_targets = rewards + (self.gamma  * Q_targets_next * (1 - dones))

        # get outputs
        self.qnetwork_local.train()
        Q_expected = self.qnetwork_local.forward(states).gather(1, actions)

        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # clear gradients
        self.optimizer.zero_grad()

        # update weights local network
        loss.backward()

        # take one SGD step
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step % self.target_update_freq == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



        