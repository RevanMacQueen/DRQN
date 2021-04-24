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


def extract_epsisode_lengths(root: str, param_names: list, filters=None, cutoffs=None, last_n=False):
    """
    Extract epsiode lengths

    args
        root : directory where results are stored
        param_names : a list of parameters to extract for each run, stored in params.json 
        env: which environment to extract this for
    """
    def cutoff(episode_len, cutoff):
        new_episode_len = []
        current_len = 0
        for i in episode_len:
            current_len += i
            if i != cutoff: 
                new_episode_len.append(current_len)
                current_len = 0

        return new_episode_len

    all_episodes = []
    all_params = []

    for dir in tqdm(root.iterdir()):    
        params_file = dir/'params.json'               
        params = json.load(open(params_file))
        params_run  = [] # relevant parameters for a single run
        env = params['env']

        filter = False
        for p in param_names:
            params_run.append(params.get(p, None))
            value = params.get(p, None)
            
            if filters is not None:
                allowed_values = filters.get(p)
                if allowed_values is not None and value not in allowed_values:
                    filter = True

        if filter == False:
            all_params.append(params_run)
            episode_len = np.load(dir/"episode_lengths.npy")

            if cutoffs is not None:
                # add cut off episodes to subsequent
                episode_len = cutoff(episode_len, cutoffs[env])

            if last_n:
                all_episodes.append(episode_len[-last_n:])
            else:
                all_episodes.append(episode_len)
    
    return np.array(all_params), np.array( all_episodes)



root = Path('results_best')
filters = {
    'buffer_size' : [10000],
    'env': ['envs:random_maze-v0', 'CartPole-v1']}

cutoffs = {
    'envs:random_maze-v0' : 1000,
    'CartPole-v1' : -1
}

params, ep_lens = extract_epsisode_lengths(root, ['model_arch', 'env', 'seed', 'learning_freq', 'target_update_freq', 'seq_len', 'learning_rate', 'buffer_size'], filters, cutoffs)


# In[80]:


df1 = pd.DataFrame(params)
df2 = pd.DataFrame(ep_lens)

print(df2)
df = pd.concat([df1, df2], axis=1)
df.columns = ['Model', 'Env', 'Seed', 'Learning Freq.', 'Target Update Freq.', 'L', 'Step Size', 'Buffer Size', 
'Episode Lengths']
df = df.drop(['Seed'],axis=1)
df['Algorithm'] = df['Model'] + ', ' + df['L'] 
df = df.drop(['Model', 'L' ], axis=1)

columns = ['Algorithm', 'Env', 'Learning Freq.', 'Target Update Freq.', 'Step Size', 'Buffer Size'] #parameters

# rename environments
df = df.replace('envs:random_maze-v0', 'Maze')

# rename Algorithms
df = df.replace('FFN, 10', 'DQN')
df = df.replace('FFN, 1', 'DQN')

df = df.replace('RNN, 1', 'DRQN, 1')
df = df.replace('RNN, 2', 'DRQN, 2')
df = df.replace('RNN, 4', 'DRQN, 4')
df = df.replace('RNN, 8', 'DRQN, 8')


# In[81]:

'''
Bins episode lengths in each run into 100 bins and then creates learning curves,
averaging across the corresponding bins for all 100 runs of the best parameter settings
'''    


envs = ['CartPole-v1', 'Maze']
for env in envs:
    algs = []
    plt.figure()
    ax = plt.gca()
    for alg in df.Algorithm.unique():
        algs.append(alg)
        df_DRQN4 = df.loc[(df['Env'] == env) & (df['Algorithm'] == alg)]
        data = df_DRQN4['Episode Lengths'].values
        n_bins = 100
        bin_size = 200000/n_bins
        bins = [[] for i in range(n_bins)]

        for run in data:
            sum_ = 0
            for i, ep_len in enumerate(run):
                ind = int(sum_ // bin_size)  # which bin to place data (based on start time)
                bins[ind].append(ep_len)
                sum_ += ep_len 

        print(np.min([len(i) for i in bins]))
        mean = np.array([ np.sum(i)/len(i) for i in bins])
        sigma = np.array([np.std(i,axis=0)/np.sqrt(len(i)) for i in bins])
        t = np.linspace(0, n_bins-1, n_bins)
        plt.plot(mean)
        ax.fill_between(t, mean+sigma, mean-sigma, alpha=0.6)

    plt.xlabel('Run Progress %')
    plt.ylabel('Average Episode Length')
    plt.title(env)
    plt.legend(algs)
    plt.show()