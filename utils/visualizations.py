from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from gym_recording_modified.playback import get_recordings
import seaborn as sns
from scipy.stats import sem

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
                ax.yscale('log')
            
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

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=False, figsize=(10,10))

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
                ax.set_yscale('log')
                
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
            col_idx += 1
        row_idx += 1

    fig.tight_layout()


def plot_avg_episode_length(params: np.array, data: np.array, categories=[], shapes=[],  plot=None, xlabel='', ylabel='', title='', rowdict={}, coldict={}, tick_fmt='', shape_fmt='', scale='linear'):

    """
    General-use function for plotting a episode length

    args: 
        params : np.array shape (num_runs, num_params) with parameters for each run
        data : np.array shape (num_runs, num_timesteps) with data
        categories : list of indices of parameters. This will determine the categorical variables plotted
        shapes : index of parameters, will plot variables in this column with different shapes
    """

    font = {'size'   : 14}

    matplotlib.rc('font', **font)

    param_categories = np.unique(params[:, categories], axis=0)
    shape_categories = np.unique(params[:, shapes], axis=0)
    bar_width = (1 / len(shape_categories))*0.9

    colours = sns.cubehelix_palette(len(shape_categories), start=.5, rot=-.75).as_hex()
    colour_dict = {}

    for i in range(len(shape_categories)):
        colour_dict[ shape_fmt % tuple(shape_categories[i, :])     ] = colours[i]

    x_pos = 0 # center for a collection of bars

    labels = []
    for param_cat in param_categories:
        bar_positions = [(x_pos-0.5)+ (i*bar_width) for i in range(len(shape_categories))]
        
        for i in range(len(shape_categories)):
            bar_pos = bar_positions[i]
            shape_cat = shape_categories[i]
            inds = np.where((params[:, categories]==param_cat).all(axis=1) & (params[:, shapes]==shape_cat).all(axis=1)    )

            if inds[0].shape[0] ==0:
                break

            #data_ =   np.concatenate(   [data[i][-50:-1] for i in inds[0] ],  axis=None)
            data_ = np.hstack(data[inds])
           
            plt.bar(bar_pos, np.mean(data_), align='edge', width = bar_width, color=colour_dict[shape_fmt % tuple(shape_cat)])
            plt.errorbar(bar_pos+(bar_width/2), np.mean(data_),yerr=sem(data_), ecolor='black', capsize=3 )
        
        labels.append(tick_fmt % tuple(param_cat))

        x_pos += 1
            
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    colour_dict['DQN'] = colour_dict.pop('FFN, 10')
    plt.legend(handles=[mpatches.Patch(color=v, label=k.replace('RNN', 'DRQN')) for (k, v) in colour_dict.items()])
    plt.xticks(np.arange(len(param_categories)), labels=labels, rotation=290)

    if scale != 'linear':
        plt.yscale('log', base=2)

    plt.tight_layout()


def rolling_sem(data,window):
    sems = []
    for i in range(0, data.shape[1]-window+1):
        sems.append(sem( data[:, i:i+window],axis=None   )  )

    sems = np.array(sems)
    return sems


def plot_rewards(params: np.array, data: np.array, row=None, col=None, plot=None, xlabel='', ylabel='', title='', rowdict={}, coldict={}, plot_fmt='',window=1, sample=1):
    """
 
    """
    plot_params = np.unique(params[:, plot], axis=0)
    colours = sns.cubehelix_palette(len(plot_params), start=.5, rot=-.75).as_hex()
    colour_dict = {}

    for i in range(len(plot_params)):
        colour_dict[plot_fmt % tuple(plot_params[i, :])     ] = colours[i]


    for plot_param in plot_params:
        inds = np.where((params[:, plot]==plot_param).all(axis=1))
        data_ = np.mean(data[inds], axis=0) # data to plot
       

        # if window == 1:
        #     error = sem(data[inds], axis=0)
        # else:
        #     error = rolling_sem(data,window)
        error = sem(data[inds], axis=0)
        data_ = np.convolve(data_, np.ones(window)/window, mode='same')




        error_up = data_ + error/2
        error_low = data_ - error/2


        data_ = data_[0::sample]
        error_up =  error_up[0::sample]
        error_low =  error_low[0::sample]
        plt.plot(data_, color=colour_dict[plot_fmt % tuple(plot_param)])
        plt.fill_between(range(data_.shape[0]), error_low, error_up, color=colour_dict[plot_fmt % tuple(plot_param)], alpha=0.4)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    colour_dict['DQN'] = colour_dict.pop('FFN, 10')
    plt.legend(handles=[mpatches.Patch(color=v, label=k.replace('RNN,', 'DRQN, L=')) for (k, v) in colour_dict.items()])
    plt.tight_layout()