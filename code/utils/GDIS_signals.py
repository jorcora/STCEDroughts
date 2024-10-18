#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY 
import numpy as np
                                
# MATPLOTLIB
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# PANDAS
import pandas as pd

def normalize_events(probs_signals):
    """Gets a dictionary where each key contains a probability vector
    Computes min - max normalization over all keys in the dictionary
    Returns the modified dictionary
    Exception incluted to account for the key -999 which includes all events
    and has to be defined separately

    :param probs_signals: signals probabilities
    :type probs_signals: np.ndarray
    :return: normalized signals probabilities across events
    :rtype: np.ndarray
    """
    # get min and max signal value in all events
    # avoid accounting for the -999. And the background, 0?
    probs_individual_events = {key: probs_signals[key] for key in probs_signals.keys() if key != -999}
    max_probs = np.nanmax(np.concatenate(list(probs_individual_events.values())))
    min_probs = np.nanmin(np.concatenate(list(probs_individual_events.values())))
    
    # min - max normalization
    probs_signals_normalized = {key: (probs_signals[key] - min_probs)/(max_probs - min_probs) 
                                for key in probs_signals.keys()}
    
    # -999
    probs_signals_normalized[-999] = (probs_signals[-999] - np.nanmin(probs_signals[-999]))/(np.nanmax(probs_signals[-999]) - np.nanmin(probs_signals[-999]))
    
    return probs_signals_normalized
    
def normalize_timesteps(probs_signals):
    """Gets a dictionary where each key contains a probability vector
    Rescales vector elements according to the ratio elements of vector vs all elements
    Returns the modified dictionary

    :param probs_signals: signals probabilities
    :type probs_signals: np.ndarray
    :return: normalized signals across time
    :rtype: np.ndarray
    """
    # Scale all events
    probs_signals_normalized = {key: [] for key in probs_signals.keys()}
    for key in probs_signals.keys():
        
        # Number of elements in all timesteps for current event
        n_total = np.sum(np.isfinite(np.concatenate(probs_signals[key])))
                
        for timestep in range(len(probs_signals[key])):
            
            # # ratio number of elements vs total number of elements in each event for all timesteps
            probs_signals_normalized[key].append(probs_signals[key][timestep] * len(probs_signals[key][timestep])/n_total)
    
    return probs_signals_normalized
    
def plot_GDIS_signals(times, probs_signals_yhat, probs_signals_labels, save_path, metric = 'mean', 
                      normalized_timesteps = False, normalized_events = True, print_format = 'png',
                      labels_as_True = True, set_legend = True, yscale = 'linear', 
                      smooth = False, smooth_type='ewm', alpha = np.nan):
    """Function to plot the signals
    """
    
    # Elements to plot and background
    elements = np.array(list(probs_signals_yhat.keys()))
    elements = elements[elements != 0]
    background = 0
    
    # Correction for the number of elements that define each timestep 
    if normalized_timesteps: 
        print('Normalizing timesteps')
        
        # Normalize each timestep correcting for the number of pixels
        # involved in that timestep and the total of pixels involved
        probs_signals_yhat = normalize_timesteps(probs_signals_yhat)
        probs_signals_labels = normalize_timesteps(probs_signals_labels)
    
    # Get the probabilites according to the metric
    probs_signals_yhat_metric = {key: [] for key in probs_signals_yhat.keys()}
    probs_signals_labels_metric = {key: [] for key in probs_signals_labels.keys()}
    
    for key in probs_signals_yhat.keys():
        for timestep in range(len(probs_signals_yhat[key])):
            
            if metric == 'mean':
                probs_signals_yhat_metric[key].append(np.nanmean(probs_signals_yhat[key][timestep]))
                probs_signals_labels_metric[key].append(np.nanmean(probs_signals_labels[key][timestep]))
    
            elif metric == 'median':
                
                probs_signals_yhat_metric[key].append(np.nanmedian(probs_signals_yhat[key][timestep]))
                probs_signals_labels_metric[key].append(np.nanmedian(probs_signals_labels[key][timestep]))
            
            else: 
                print('No valid metric')

    # Normalize according to maximum value in all events
    if normalized_events: 
        print('Normalizing events')
        
        probs_signals_yhat_metric = normalize_events(probs_signals_yhat_metric)
        probs_signals_labels_metric = normalize_events(probs_signals_labels_metric)
        
    if labels_as_True:
        
        # Sets the labels to 1 if there is at least one pixel true. Ignores normalization of labels and the mean
        print('Labels for timestep set to one if one True pixel exists')
        probs_signals_labels_metric = {key: (probs_signals_labels_metric[key] > 0) for key in probs_signals_labels_metric.keys()}
    
    if smooth: 
        if smooth_type == 'ewm':
            for ikey in np.array(list(probs_signals_yhat.keys())):
                df = pd.DataFrame(data = np.array(probs_signals_yhat_metric[ikey]))
                df = df.ewm(alpha = alpha).mean()
                probs_signals_yhat_metric[ikey] = df.to_numpy().ravel()

    # Plot signal for each element
    for event in elements:
        fig = plt.figure(figsize=(10, 8))
        plt.plot(times, probs_signals_labels_metric[event], color='#8D8D8D', linestyle='-', label='Drought GT', linewidth=4.5)
        plt.plot(times, probs_signals_yhat_metric[event], color='#FF7C4C', linestyle='-', label='Drought signal', linewidth=4.5)
        plt.plot(times, probs_signals_yhat_metric[background], color='#9D7CDE', linestyle='-', label='Background signal for all Europe', linewidth=4.5)
        #plt.grid(True, axis = 'y', linestyle='--', alpha = 0.5)
        #ax = plt.gca()
        #ax.spines[['top', 'right']].set_visible(False) 
        plt.yscale(yscale)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.ylabel('$\overline{Y}(t)$', fontsize = 20)    
        if set_legend:
            plt.legend(fontsize = 18)
        plt.gcf().autofmt_xdate()
        plt.savefig((save_path + f'/Signals_{metric}_' + str(np.round(event)) + 
                    '_smooth_' + str(smooth) + '_type_' + smooth_type + '_alpha_' + str(alpha) +
                    f'.{print_format}'), 
                    format = print_format, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        plt.close(fig = fig)

