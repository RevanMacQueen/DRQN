from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import os
import time
import json
import argparse
import copy
import torch

import numpy as np

import gym
from gym_recording_modified.wrappers import TraceRecordingWrapper
import matplotlib.pyplot as plt

from agent.agent import Agent
from agent.tabular_agent import TabularAgent
from agent.settings import device
from envs.random_maze import RandomMaze
from utils.utils import to_command


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for RNN agent and environment')

    parser.add_argument('--seed', default=1, type=int, nargs='?')

    parser.add_argument('--env', default='envs:random_maze-v0', type=str)
    
    parser.add_argument('--model_arch', default='RNN', type=str, help="Type of agent. Options: RNN, FFN, tabular")

    parser.add_argument('--buffer_size', default=10000, type=int)

    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--learning_rate', default=5e-4, type=float)
    
    parser.add_argument('--epsilon', default=0.1, type=float)

    parser.add_argument('--decay', default=0.99, type=float, help="Scalar which determines how much to decay epsilon each episode")

    parser.add_argument('--min_epsilon', default=0.1, type=float)

    parser.add_argument('--hidden_layer_size', default=64, type=int)  

    parser.add_argument('--num_layers', default=1, type=int)  

    parser.add_argument('--seq_len', default=10, type=int)  

    parser.add_argument('--learning_starts', default=50000, type=int)

    parser.add_argument('--learning_freq', default=1, type=int)

    parser.add_argument('--target_update_freq', default=1, type=int)

    parser.add_argument('--gamma', default=0.99, type=float)

    parser.add_argument('--tau', default=1e-3, type=float) 
    
    parser.add_argument('--save_path', default='results', type=str)

    parser.add_argument('--save_recording', default=False, action='store_true', help = "Whether to record interactions" )

    parser.add_argument('--only_reward', default=False, action='store_true', help = "Whether to only record rewards" )

    parser.add_argument('--show_pbar', default=False, type=bool, help = "Whether to show progress bar" )

    parser.add_argument('--n', default=5, type=int, help="size of random maze")

    parser.add_argument('--cycles', default=3, type=int, help="number of cycles in random maze")

    parser.add_argument('--num_iterations', default=10**6, type=int)
      
    parser.add_argument('--iterate', default='steps', type=str, help= "Whether max number of iterations is determined by episodes or timesteps")

    parser.add_argument('--state_representation', default='flat_grid', type=str, help='How to represent the state')

    parser.add_argument('--run_dir', default='auto')

    

    return vars(parser.parse_args())

def main(args):
    
    if args['run_dir'] == 'auto':
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        save_dir = Path(args['save_path'])/ str(timestamp)
    else:
        save_dir = Path(args['save_path'])/  args['run_dir']
    
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
    
    #env.showPNG()
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

    agent_args = copy.deepcopy(args)
    agent_args['input_dim'] = ob_dim
    agent_args['action_dim'] = ac_dim

    if args['model_arch'] == 'tabular':
        agent = TabularAgent(agent_args)
    else:
        agent = Agent(agent_args)

    step_counter = 0 # counter for length of episodes 
    episode_lengths = [] # list of episode length
    start_time = time.time()
    obs = env.reset()


    num_iterations = args['num_iterations']

    if args['show_pbar']:
        pbar = tqdm(total=num_iterations)
    

    itr = 0
    episode = 0
    terminate = False
    while not terminate :
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.train_step(obs, action, reward, next_obs, done)
        obs = next_obs

        step_counter += 1
        if done: # end of episode
            episode_lengths.append(step_counter)
            episode += 1 
            step_counter = 0
            obs = env.reset()
            if args['iterate'] == 'episodes':
                if episode >=  num_iterations:
                    terminate = True
                if args['show_pbar']:
                    pbar.update(1)

        itr +=1
        if args['iterate'] == 'steps':
            if itr >=  num_iterations:
                terminate = True
            if args['show_pbar']:
                pbar.update(1)
    
    if len(episode_lengths) ==0: # incase nothing got added
        episode_lengths.append(step_counter) 

    env.close()
    end_time = time.time()
    np.save(save_dir/'episode_lengths.npy', episode_lengths )
    np.save(save_dir/'time.npy', np.array(end_time-start_time))

    plt.plot(episode_lengths)
    plt.show()

    # log experiments that finished 
    with open('experiments_done.txt', 'a') as output:
        output.write(to_command(args))

if __name__ == '__main__':
    args = get_args()
    main(args)

