# script for running experiments with best parameters

import numpy as np
import argparse
from main import main
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
import itertools

from utils.utils import to_command

def get_args(): 
    """
    This function will extract the arguments from the command line
    """
 
    parser = argparse.ArgumentParser(description='Experiments for RNN project')

    parser.add_argument('--output_type',  default= "compute_canada_format", type=str, choices=("bash_file", "compute_canada_format", "execute"), help="What should be the output of this file: bash_file: generates a bash file of the commands that you should run in linux for all the experiments | compute_canada_format: generates a file that can be used to run all the experiments on compute canada | execute: will run all experiments on your computer")

    parser.add_argument('--output_path', default='experiments_ablation', type=str,
            nargs='?', help="The path to save the output file of this script")

    parser.add_argument('--save_path', default='results', type=str,
            nargs='?', help="The root path that should be used to save the results of our experiments (This path will be passed to main.py as an argument)")

    parser.add_argument('--num_threads', default=1, type=int,
            nargs='?', help="How many concurrent experiments to run")

    return vars(parser.parse_args())


### DEFINE ARGUMENTS ###

# general arguments
GENERAL_ARGS = {
    'seed' : 1,
    'save_path': 'results',
    'run_dir': 'auto',
    'save_recording' : True,
    'only_reward' : True,
    'show_pbar' : False,
    'env' : 'envs:random_maze-v0'
   }


# default Arguments for RNN
ZERO_RNN_ARGS = {
    'model_arch' : 'RNN',
    'buffer_size' : 10000,
    'batch_size' : 64,
    'hidden_layer_size' : 64,
    'num_layers' : 0,
    'seq_len' : 1,
    'tau' : 1,
    'learning_starts': 100,
    'learning_freq' : 100,
    'target_update_freq' : 1,
    'learning_rate' : 5e-4,
    'epsilon': 1, # all methods use epsilon greedy 
    'min_epsilon' : 0.01,
    'decay' : 0.995,
    'state_representation' : 'one_hot',
    'buffer' : 'episodes',
    'zero_state' : True
    }

# default Arguments for RNN
RNN_ARGS = {
    'model_arch' : 'RNN',
    'buffer_size' : 10000,
    'batch_size' : 64,
    'hidden_layer_size' : 64,
    'num_layers' : 0,
    'seq_len' : 1,
    'tau' : 1,
    'learning_starts': 100,
    'learning_freq' : 100,
    'target_update_freq' : 1,
    'learning_rate' : 5e-4,
    'epsilon': 1, # all methods use epsilon greedy 
    'min_epsilon' : 0.01,
    'decay' : 0.995,
    'state_representation' : 'one_hot',
    'buffer' : 'episodes'
    }

# default Arguments for FFN with modified buffer
FFN_MB_ARGS = {
    'model_arch' : 'FFN',
    'buffer_size' : 10000,
    'batch_size' : 64,
    'hidden_layer_size' : 64,
    'num_layers' : 2,
    'tau' : 1,
    'learning_starts': 100,
    'learning_freq' : 100,
    'target_update_freq' : 1,
    'learning_rate' : 5e-4,
    'epsilon': 1, # all methods use epsilon greedy
    'min_epsilon' : 0.01,
    'decay' : 0.995,
    'state_representation' : 'one_hot',
    'buffer' : 'episodes',
    'seq_len' : 1
    }

FFN_ARGS = {
    'model_arch' : 'FFN',
    'buffer_size' : 10000,
    'batch_size' : 64,
    'hidden_layer_size' : 64,
    'num_layers' : 2,
    'tau' : 1,
    'learning_starts': 100,
    'learning_freq' : 100,
    'target_update_freq' : 1,
    'learning_rate' : 5e-4,
    'epsilon': 1, # all methods use epsilon greedy
    'min_epsilon' : 0.01,
    'decay' : 0.995,
    'state_representation' : 'one_hot',
    'buffer' : 'steps',
    'seq_len' : 1
    }

# Arguments for tabular
TAB_ARGS = {
    'model_arch' : 'tabular',
    'learning_rate' : 5e-4,
    'epsilon': 0.1, # all methods use epsilon greedy 
    'state_representation' : 'integer'
    }

MODEL_ARGS = {
    'FFN' : FFN_ARGS,
    'RNN' : RNN_ARGS,
    'tabular' : TAB_ARGS
    }

# specific to maze environment
MAZE_ARGS = {
    'n' : 5,
    'cycles' : 3,
    'gamma' : 0.95,
    'num_iterations' : 200000,
    'iterate' : 'steps',
    }

CARTPOLE_ARGS = {
    'num_iterations' : 200000,
    'gamma' : 0.99,
    'iterate' : 'steps',
}

MOUNTAINCAR_ARGS = {
    'num_iterations' : 200000,
    'gamma' : 1,
    'iterate' : 'steps',
}

ENV_ARGS = {
    'envs:random_maze-v0' : MAZE_ARGS,
    'CartPole-v1' : CARTPOLE_ARGS, # no extra arguments needed
    'MountainCar-v0' : MOUNTAINCAR_ARGS, # no extra arguments needed
    'MountainCar1000-v0' : MOUNTAINCAR_ARGS # no extra arguments needed
}

### Experimental Parameters ###
np.random.seed(569)
SEEDS = np.random.randint(0, 10000, size=30)

# determined though new_visualizations script
BEST_PARAMS = [
    ('FFN', '1', 'envs:random_maze-v0', '1', '100', '0.0005', '10000'),
    ('RNN', '1', 'envs:random_maze-v0', '10', '100', '0.005', '10000'),
    ('RNN', '2', 'envs:random_maze-v0', '1', '100', '0.0005', '10000'),
    ('RNN', '4', 'envs:random_maze-v0', '1', '100', '5e-05', '10000'),
    ('RNN', '8', 'envs:random_maze-v0', '1', '1', '5e-05', '10000')]
    # ('FFN', '1', 'CartPole-v1', '1', '100', '0.0005', '10000'),
    # ('RNN', '1', 'CartPole-v1', '10', '100', '0.005', '10000'),
    # ('RNN', '2', 'CartPole-v1', '10', '1000', '0.005', '10000'),
    # ('RNN', '4', 'CartPole-v1', '1', '100', '5e-05', '10000')]
    #('RNN', '8', 'CartPole-v1', '10', '1000', '0.005', '10000')] same as above


ARG_MAP = {
    'FFN' : FFN_ARGS,
    'RNN' : RNN_ARGS,
    'FFN_MB' : FFN_MB_ARGS,
    'ZERO_RNN' : ZERO_RNN_ARGS
}

def experiments(script_args):
    run_num = 0
    all_args = []
    bash_file_commands = []

    for i in BEST_PARAMS:

        if i[2] == 'envs:random_maze-v0':
            algs =  ['FFN', 'RNN', 'FFN_MB', 'ZERO_RNN']
        else:
            algs =  ['RNN']
            
        for alg in algs:
            for seq_len in [1, 2, 4, 8]:
                for seed in SEEDS:
                    model_args = deepcopy(ARG_MAP[alg])
                    
                    general_args = deepcopy(GENERAL_ARGS)
                    general_args['seed'] = int(seed) # int needed to save to json; numpy int32 raises error
                    general_args['save_path'] = script_args['save_path']
                    general_args['env'] = i[2]
                    general_args['run_dir'] = str(run_num)

                    model_args['model_arch'] = alg
                    model_args['seq_len'] = seq_len

                    model_args['learning_freq'] = i[3]
                    model_args['target_update_freq'] = i[4]
                    model_args['learning_rate'] = i[5]
                    model_args['buffer_size'] = i[6]

                    run_args = {**general_args, **model_args, **ENV_ARGS[i[2]]}  # combine dictionaries

                    if script_args['output_type'] == 'execute':                     
                        if script_args['num_threads'] == 1:
                            main(run_args)
                    
                        elif script_args['num_threads'] != 1:
                            all_args.append(run_args)
                    
                    else: 
                        bash_file_commands.append(to_command(run_args))

                    run_num +=1

    # run multithreaded experiments
    if script_args['output_type'] == 'execute' and  script_args['num_threads'] != 1:
        with Pool(script_args['num_threads']) as p:
            
            r = list(tqdm(p.imap(main, all_args), total=len(all_args)))

    if script_args['output_type'] == 'bash_file':
        with open(script_args['output_path'] + '.bash', 'w') as output:
            for row in bash_file_commands:
                output.write(str(row))

    elif script_args['output_type'] == 'compute_canada_format':
        with open(script_args['output_path'] + '.txt', 'w') as output: # This .txt file can use a command list for GNU Parallel
            for row in bash_file_commands:
                output.write(str(row))
 

if __name__ == '__main__':
    ARGS = get_args()
    experiments(ARGS)

