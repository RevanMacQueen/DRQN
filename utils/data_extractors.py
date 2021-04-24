from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from gym_recording_modified.playback import get_recordings
from tqdm import tqdm


def extract_episode_lengths(root: str, param_names: list, filters=None, cutoffs=None, last_n=False):
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

def extract_num_episodes(root: str, param_names: list, filters=None, cutoffs=None):
    """
    Extract number of episodes

    args
        root : directory where results are stored
        param_names : a list of parameters to extract for each run, stored in params.json 
        env: which environment to extract this for
    """

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
                episodes_completed = np.sum(episode_len != cutoffs[env])
            else:
                episodes_completed  = episode_len.shape[0]

            all_episodes.append(episodes_completed)
    
    return np.array(all_params),  np.array( all_episodes)

def extract_rewards(root: str, param_names: list, filters=None):
    """
    Extract rewards

    args
        root : directory where results are stored
        param_names : a list of parameters to extract for each run, stored in params.json 
    """

    all_rewards = []
    all_params = []

    for dir in root.iterdir():    
        params_file = dir/'params.json'               
        params = json.load(open(params_file))
        params_run  = [] # relevant parameters for a single run
        
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
            rewards = get_recordings(dir, extract='rewards')['rewards']
            all_rewards.append(rewards)
        
    return np.array(all_params),  np.array(all_rewards)


def extract_times(root: str, param_names: list, filters=None):
    """
    Extract time for each run

    args
        root : directory where results are stored
        param_names : a list of parameters to extract for each run, stored in params.json 
    """

    all_times = []
    all_params = []

    for dir in root.iterdir():    
        params_file = dir/'params.json'               
        params = json.load(open(params_file))
        params_run  = [] # relevant parameters for a single run
        
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


            time = np.load(dir/"time.npy")
            all_times.append(time)
        
    return np.array(all_params),  np.array(all_times)