from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from gym_recording_modified.playback import get_recordings

def extract_epsisode_lengths(root: str, param_names: list,):
    """
    Extract epsiode lengths

    args
        root : directory where results are stored
        param_names : a list of parameters to extract for each run, stored in params.json 
        env: which environment to extract this for
    """

    all_episodes = []
    all_params = []

    for dir in root.iterdir():    
        params_file = dir/'params.json'               
        params = json.load(open(params_file))

        params_run  = [] # relevant parameters for a single run
        for p in param_names:
            params_run.append(params.get(p, None))

        all_params.append(params_run)
        episode_len= np.load(dir/"episode_lengths.npy")
        all_episodes.append(episode_len)

        
    
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
            allowed_values = filters.get(p)
            
            if filter is not None:
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
            allowed_values = filters.get(p)
            
            if filter is not None:
                if allowed_values is not None and value not in allowed_values:
                    filter = True

        if filter == False:
            all_params.append(params_run)


            time = np.load(dir/"time.npy")
            all_times.append(time)
        
    return np.array(all_params),  np.array(all_times)