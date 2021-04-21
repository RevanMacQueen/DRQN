from pathlib import Path
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gym_recording_modified.playback import get_recordings
from utils.data_extractors import *
from utils.visualizations import *
import seaborn as sns
from scipy.stats import sem
from scipy.signal import convolve2d


# learning curves
root = Path('results_best')
filters = None
cutoffs = {
    'envs:random_maze-v0' : 1000,
    'CartPole-v1' : -1
}

params, data = extract_episode_lengths(root, ['model_arch', 'env', 'seq_len'], filters, cutoffs)

def transform(row):
    completed = np.zeros(200001)
    completed[np.cumsum(np.array(row))] = 1
    completed = np.cumsum(completed)
    return completed




font = {'size'   : 14}
matplotlib.rc('font', **font)
algs =  np.unique(params[:, 0])
envs = np.unique(params[:, 1])
seq_lens = np.unique(params[:, 2])
colours = sns.cubehelix_palette(5, start=.5, rot=-.75).as_hex()

for env in envs:
    colour_idx = 0
    for alg in ['FFN', 'RNN']:
        for seq_len in ['1','2','4','8']:

            inds = np.where((params[:, 1] == env) & (params[:, 0]==alg) & (params[:, 2]==seq_len))
      
            if len(inds[0]) == 0:
                break

            data_ = data[inds]
            params_ = data[inds]
            
            data_trans = np.zeros([100, 200001])

            for i in range(data_.shape[0]):
                data_trans[i, :] = transform(data_[i]) 

            x = np.mean(data_trans, axis =0)
            err = sem(data_trans, axis=0)

            err_up = x + err/2
            err_low = x - err/2
            
            sample = 100

            y = x[0::sample]
            
            x = np.arange(x.shape[0])
            x = x[0::sample]
            
            err_up =  err_up[0::sample]
            err_low =  err_low[0::sample]

            plt.plot(x, y, color=colours[colour_idx], label= alg + ', ' + seq_len)
            plt.fill_between(x, err_low, err_up, color=colours[colour_idx], alpha=0.4)
            colour_idx+=1

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Step')
    plt.ylabel('Number of episodes completed')
    plt.legend(loc='upper left')
    plt.tight_layout()

    env_rename = {'envs:random_maze-v0':'maze'}
    plt.savefig('figures/'+ env_rename.get(env, env) + '_learning_curve.pdf', bbox_inches='tight')   
 
    plt.clf()


