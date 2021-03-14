#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:17:55 2021

@author: kerrick
"""
import numpy as np
import matplotlib.pyplot as plt

results_dir = '/home/kerrick/uAlberta/Winter2021/Empirical/Empirical_RL_RNN-main/results/1615746659.854553'
rets = np.load(results_dir+'/episode_rets.npy')
successes = rets > -1
window = 10
success_freq = []
i = window
while i < len(rets):
    success_freq.append(np.sum(successes[i-window:i])/window)
    i += 1
    
plt.plot(success_freq)