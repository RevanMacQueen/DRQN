"""
Script for generating an array of experiments. Can either be run local (single or multithreaded) or on compute canada
"""

"""
Generates tabular experiments. Can either run them using this the --execute flag or output to a .txt file
"""

import numpy as np
import argparse
from main import main 
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool

# general arguments
ARGS = {
    'seed' : 1
    'save_path': 'results',
    'save_recording' : True,
    'only_reward' : True,
   }


# Arguments for RNN
RNN_ARGS = {
    'model_arch' : 'RNN',
    'buffer_size' : 10000,
    'batch_size' : 64,
    'hidden_layer_size' : 64,
    'num_layers' : 1,
    'seq_len' : 10,
    'tau' : 1e-3,
    'learning_starts': 50000,
    'learning_freq' : 1,
    'target_update_freq' : 1,
    'learning_rate' : 5e-4,
    'epsilon': 0.1 # all methods use epsilon greedy 
    }

# Arguments for FFN
FFN_ARGS = {
    'model_arch' : 'FFN',
    'buffer_size' : 10000,
    'batch_size' : 64,
    'hidden_layer_size' : 64,
    'num_layers' : 1,
    'tau' : 1e-3,
    'learning_starts': 50000,
    'learning_freq' : 1,
    'target_update_freq' : 1,
    'learning_rate' : 5e-4,
    'epsilon': 0.1 # all methods use epsilon greedy
    }

# Arguments for Tabular
TAB_ARGS = {
    'model_arch' : 'DQN',
    'learning_rate' : 5e-4,
    'epsilon': 0.1 # all methods use epsilon greedy 
    }

MODEL_ARGS = {
    'FFN' : FFN_ARGS,
    'RNN' : RNN_ARGS,
    'Tabular' : TAB_ARGS
    }

# specific to maze environment
MAZE_ARGS = {
    'n' : 5,
    'cycles' : 3,
    'gamma' : 0.95,
    'num_iterations' : 100000
    }

ENV_ARGS = {
    'envs:random_maze-v0' : MAZE_ARGS
}

### Experimental Parameters ###
np.random.seed(569)
SEEDS = np.random.randint(0, 10000, size=30)
MODELS = ['FFN', 'RNN', 'tabular']
ENV_IDS = ['envs:random_maze-v0'] 
RUNS = {
    'FFN' : ffn_runs,
    'RNN' : rnn_runs,
    'Tabular' : tab_runs
}
###############################


def ffn_runs():
    """
    Returns a list of all different ffn configurations to run 
    """
    return [FFN_ARGS]

def rnn_runs():
    """
    Returns a list of all different rnn configurations to run 
    """
    return [RNN_ARGS]

def tab_runs():
    """
    Returns a list of all different rnn configurations to run 
    """
    return [TAB_ARGS]


def to_command(dic):
    command = 'python3 main.py'
    for key, value in dic.items():
        if key == "only_rewards" :
            command += ' --{}'.format(key)
        else:
            command += ' --{} {}'.format(key, value)

    return command + '\n'

def get_args(): 
    """
    This function will extract the arguments from the command line
    """
 
    parser = argparse.ArgumentParser(description='Experiments for RNN project')

    parser.add_argument('--output_type',  default= "compute_canada_format", type=str, choices=("bash_file", "compute_canada_format", "execute"), help="What should be the output of this file: bash_file: generates a bash file of the commands that you should run in linux for all the experiments | compute_canada_format: generates a file that can be used to run all the experiments on compute canada | execute: will run all experiments on your computer")

    parser.add_argument('--output_path', default='experiments', type=str,
            nargs='?', help="The path to save the output file of this script")

    parser.add_argument('--save_path', default='results', type=str,
            nargs='?', help="The root path that should be used to save the results of our experiments (This path will be passed to main.py as an argument)")

    parser.add_argument('--num_threads', default=1, type=int,
            nargs='?', help="How many concurrent experiments to run")

    return vars(parser.parse_args())


def main(script_args):

    # Count number of runs
    num_runs = 0
    for env_id in ENV_IDS:
        for model in MODELS:
           for seed in SEEDS:
                model_args = RUNS[model].()
                for model_arg in model_args:        
                    num_runs+=1

    if script_args['num_threads'] == 1:
        pbar = tqdm(total=num_runs)

    all_args = []
    bash_file_commands = []

     for env_id in ENV_IDS:
        for model in MODELS:
           for seed in SEEDS:
                all_model_args = RUNS[model].()
                for model_args in all_model_args:        
                    
                    general_args = deepcopy(ARGS)
                    general_args['seed'] = seed
                    general_args['save_path'] = script_args['save_path']

                    run_args = general_args | model_args | ENV_ARGS[env_id] # combine dictionaries


                    if args['output_type'] == 'bash_file':
                        bash_file_commands.append(to_command(run_args))
                    
                    elif args['output_type'] == 'execute' and args['num_threads'] == 1:
                        main_tabular(run_args)
                        with open('experiments_done_so_far.txt', 'w+') as output:
                            output.write(to_command(run_args))
                    
                    elif args['output_type'] == 'execute' and args['num_threads'] != 1:
                        all_args.append(run_args)
                    
                    elif args['output_type'] == 'compute_canada_format':
                        bash_file_commands.append(to_command(run_args))

                    if args['num_threads'] == 1:
                        pbar.update(1)

    # run multithreaded experiments
    if args['output_type'] == 'execute' and  args['num_threads'] != 1:
        with Pool(args['num_threads']) as p:
            r = list(tqdm(p.imap(main_tabular, all_args), total=len(all_args)))

    if args['output_type'] == 'bash_file':
        with open(args['output_path'] + '.bash', 'w') as output:
            for row in bash_file_commands:
                output.write(str(row))
    elif args['output_type'] == 'compute_canada_format':
        with open(args['output_path'] + '.txt', 'w') as output: # This .txt file can use a command list for GNU Parallel
            for row in bash_file_commands:
                output.write(str(row))
 

if __name__ == '__main__':
    ARGS = get_args()
    main(ARGS)
