from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import os
import json
import argparse

import numpy as np

import gym
from gym_recording_modified.wrappers import TraceRecordingWrapper
import matplotlib.pyplot as plt

from agent.agent import Agent
from agent.tabular_agent import TabularAgent
from envs.random_maze import RandomMaze

def get_args():
    parser = argparse.ArgumentParser(description='Parameters for RNN agent and environment')

    parser.add_argument('--seed', default=1, type=int, nargs='?')

    parser.add_argument('--env', default='envs:random_maze-v0', type=str)
    
    parser.add_argument('--model_arch', default='RNN', type=str, help="Type of agent. Options: RNN, FFN, tabular")

    parser.add_argument('--buffer_size', default=10000, type=int)

    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--learning_rate', default=5e-4, type=float)
    
    parser.add_argument('--epsilon', default=0.1, type=float)

    parser.add_argument('--hidden_layer_size', default=64, type=int)  

    parser.add_argument('--num_layers', default=1, type=int)  

    parser.add_argument('--seq_len', default=10, type=int)  

    parser.add_argument('--learning_starts', default=50000, type=int)

    parser.add_argument('--learning_freq', default=1, type=int)

    parser.add_argument('--target_update_freq', default=1, type=int)

    parser.add_argument('--gamma', default=0.95, type=float)

    parser.add_argument('--tau', default=1e-3, type=float) 
    
    parser.add_argument('--save_path', default='results', type=str)

    parser.add_argument('--save_recording', default=False, action='store_true', help = "Whether to record interactions" )

    parser.add_argument('--only_reward', default=False, action='store_true', help = "Whether to only record rewards" )

    parser.add_argument('--n', default=5, type=int, help="size of random maze")

    parser.add_argument('--cycles', default=3, type=int, help="number of cycles in random maze")

    parser.add_argument('--num_iterations', default=10**6, type=int)

    parser.add_argument('--state_representation', default='flat_grid', type=str, help='How to represent the state')

    return vars(parser.parse_args())

def main(args):
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    save_dir = Path(args['save_path'])/str(timestamp)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args['env'] == 'envs:random_maze-v0':
        env = RandomMaze(args['n'], args['cycles'], args['seed'], state_representation=args['state_representation'])
    else:
        env = gym.make(args['env'])

    if args['save_recording']:
        env = TraceRecordingWrapper(env, save_dir, only_reward=args['only_reward'])

    with open(save_dir/'params.json', 'w') as fp:
        json.dump(args, fp)
    
    seed = args['seed']
    np.random.seed(seed)
    env.seed(seed)

    # Observation and action sizes
    img = False
    if len(env.observation_space.shape) == 0: # discrete state 
        ob_dim = env.observation_space.n # this is actually the number of states
    elif not img:
        ob_dim = int(np.prod(env.observation_space.shape))
    else:
        ob_dim = env.observation_space.shape 
        

    ac_dim = env.action_space.n 

    args['input_dim'] = ob_dim
    args['action_dim'] = ac_dim


    if args['model_arch'] == 'tabular':
        agent = TabularAgent(args)
    else:
        agent = Agent(args)

    obs = env.reset()
    for i in tqdm(range(args['num_iterations'])):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.train_step(obs, action, reward, next_obs, done)

        obs = next_obs
        if done: # end of episode
            obs = env.reset()
    
    env.close()

if __name__ == '__main__':
    args = get_args()
    main(args)
