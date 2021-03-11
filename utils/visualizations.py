from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from gym_recording_modified.playback import get_recordings

def plot_timesteps(params: np.array, data: np.array, row=None, col=None, plot=None, xlabel='', ylabel='', rowdict={}, coldict={}):
    """
    General-use function for plotting a variable over time

    args: 
        params : np.array shape (num_runs, num_params) with parameters for each run
        data : np.array shape (num_runs, num_timesteps) with data
        row : index of parameters, this function will generate a subplot for each unique setting
        col : index of parameters, this function Will generate a subplot for each unique setting
        plot : index of parameters, for which a new line will be added to a subplot for each unique value
        
    """

    nrows = np.unique(params[:, row]).shape[0]
    ncols = np.unique(params[:, col]).shape[0]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=False)

    row_idx = 0
    for row_param in np.unique(params[:, row]):
        col_idx = 0
        for col_param in np.unique(params[:, col]):
            for plot_param in np.unique(params[:, plot]):
                inds = np.where(
                    (params[:, row]==row_param) & 
                    (params[:, col]==col_param) & 
                    (params[:, plot]==plot_param))
                

                data_ = np.mean(data[inds], axis=0) # data to plot
                
                ax = axs[row_idx, col_idx]
                ax.plot(data_)
                ax.set_title('%s, %s' % (rowdict.get(row_param, row_param), coldict.get(col_param, col_param)))
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            
            col_idx += 1
        row_idx += 1

    fig.tight_layout()


def plot_episode_len(params: np.array, data: np.array, row=None, col=None, plot=None, xlabel='', ylabel='', rowdict={}, coldict={}):
    """
    General-use function for plotting a episode length

    args: 
        params : np.array shape (num_runs, num_params) with parameters for each run
        data : np.array shape (num_runs, num_timesteps) with data
        row : index of parameters, this function will generate a subplot for each unique setting
        col : index of parameters, this function Will generate a subplot for each unique setting
        plot : index of parameters, for which a new line will be added to a subplot for each unique value
        
    """

    nrows = np.unique(params[:, row]).shape[0]
    ncols = np.unique(params[:, col]).shape[0]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=False)

    row_idx = 0
    for row_param in np.unique(params[:, row]):
        col_idx = 0
        for col_param in np.unique(params[:, col]):
            for plot_param in np.unique(params[:, plot]):
                inds = np.where(
                    (params[:, row]==row_param) & 
                    (params[:, col]==col_param) & 
                    (params[:, plot]==plot_param))
                
                ax = axs[row_idx, col_idx]
                for run_data in data[inds]:
                    if len(run_data) > 1:
                        ax.plot(np.arange(len(run_data)),  run_data)
                    else:
                        ax.scatter([0], run_data)
                   

                ax.set_title('%s, %s' % (rowdict.get(row_param, row_param), coldict.get(col_param, col_param)))
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
            col_idx += 1
        row_idx += 1

    fig.tight_layout()