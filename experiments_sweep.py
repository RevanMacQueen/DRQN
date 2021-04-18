"""
Script for generating an array of experiments. Can either be run local (single or multithreaded) or on compute canada
"""


import numpy as np
import argparse
from main import main
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
import itertools

from utils.utils import to_command

def get_arg_list(params, constraints, init_args):
    """
    Returns a list of dictionaries of hyperparameter settings
    """

    def satisfies_constraints(setting):
        satisfies = True
        for c in constraints:
            if c(setting) == False:
                return False
        return True

    product = itertools.product(*[p[1] for p in params])

    arg_list = []
    for setting in product: # for each setting of parameters
        args = deepcopy(init_args)

        if satisfies_constraints(setting):
            for param_idx in range(len(setting)):
                param_name = params[param_idx][0]
                args[param_name] = setting[param_idx]
            arg_list.append(args)

    return arg_list

def ffn_runs(env):
    """
    Returns a list of all different ffn configurations to run 
    """

    # different parameters to vary, first entry is name of parameter, second is possible values

    params = [
        ['learning_freq', [1, 10, 100]],
        ['target_update_freq', [1, 10, 100, 1000]],
        ['learning_rate', [5e-3, 5e-4, 5e-5]],
        ['buffer_size', [10000]]
    ]


    constraints = [
        lambda setting: setting[0] <= setting[1] # enforces that learning_freq <= target_update_freq
        ]


    return get_arg_list(params, constraints, FFN_ARGS)

def rnn_runs(env):
    """
    Returns a list of all different rnn configurations to run 
    """
    params = [
        ['learning_freq', [1, 10, 100]],
        ['target_update_freq', [1, 10, 100, 1000]],
        ['seq_len', [1, 2, 4, 8]],
        ['learning_rate', [5e-3, 5e-4, 5e-5]],
        ['buffer_size', [10000]]
    ]

    constraints = [
        lambda setting: setting[0] <= setting[1] # enforces that learning_freq <= target_update_freq
        ]


    return get_arg_list(params, constraints, RNN_ARGS)


def tab_runs(env):
    """
    Returns a list of all different rnn configurations to run 
    """
    return [TAB_ARGS]


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

# default Arguments for FFN
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
SEEDS = np.random.randint(0, 10000, size=30)[10:]
MODELS = ['FFN', 'RNN']
ENV_IDS = ['envs:random_maze-v0', 'CartPole-v1'] 

RUNS = {
    'FFN' : ffn_runs,
    'RNN' : rnn_runs,
    'tabular' : tab_runs
}
###############################

def experiments(script_args):
    # Count number of runs
    num_runs = 0
    for env_id in ENV_IDS:
        for model in MODELS:
           for seed in SEEDS:
                model_args = RUNS[model](env_id)
                for model_arg in model_args:        
                    num_runs+=1

    if script_args['num_threads'] == 1:
        pbar = tqdm(total=num_runs)

    all_args = []
    bash_file_commands = []

    run_num = 10800

    for env_id in ENV_IDS:
        for model in MODELS:
           for seed in SEEDS:
                all_model_args = RUNS[model](env_id)
                for model_args in all_model_args:        
                    
                    general_args = deepcopy(GENERAL_ARGS)
                    general_args['seed'] = int(seed) # int needed to save to json; numpy int32 raises error
                    general_args['save_path'] = script_args['save_path']
                    general_args['env'] = env_id
                    general_args['run_dir'] = str(run_num)

                    run_args = {**general_args, **model_args, **ENV_ARGS[env_id]}  # combine dictionaries

                    if script_args['output_type'] == 'execute':                     
                        if script_args['num_threads'] == 1:
                            main(run_args)
                    
                        elif script_args['num_threads'] != 1:
                            all_args.append(run_args)
                    
                    else: 
                        bash_file_commands.append(to_command(run_args))

                    if script_args['num_threads'] == 1:
                        pbar.update(1)

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
