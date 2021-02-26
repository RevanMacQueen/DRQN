import numpy as np
from agent.replay_buffer import RNNReplayBuffer

buffer_size = 100 
batch_size = 16
action_size = 2
state_size = 8
seq_len = 8
episode_len = 16
seed = 10

def setup_rnn_replay_buffer():

    rb = RNNReplayBuffer(action_size, buffer_size, batch_size, seq_len, seed)

    for i in range(buffer_size):
        rb.add_episode()
        for j in range(episode_len):
            state = i * np.ones(state_size)
            action = i * np.ones(action_size)
            reward = i
            next_state = i * np.ones(state_size)
            done = True if j == (episode_len-1) else False
            rb.add(state, action, reward, next_state, done)
    assert len(rb.memory) == buffer_size
    return rb

def test_rnn_sample():
    print("test_rnn_sample")
    rb = setup_rnn_replay_buffer()
    for i in range(10):
        states, actions, rewards, next_states, dones = rb.sample()
        assert states.shape == (batch_size, seq_len, state_size)
        assert actions.shape == (batch_size, seq_len, action_size)
        assert rewards.shape == (batch_size, seq_len, 1)
        assert next_states.shape == (batch_size, seq_len, state_size)
        assert dones.shape == (batch_size, seq_len, 1)
    print("Passed!")