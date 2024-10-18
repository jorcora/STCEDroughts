#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY
import numpy as np

# SKLEARN 
from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score, 
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

# MATPLOTLIB
import matplotlib.pyplot as plt

# TORCH
import torch
from torchmetrics.functional.classification import binary_calibration_error

# PICKLE
import pickle

# SEABORN
import seaborn as sns

# PANDAS
import pandas as pd

# UTILS
from utils import cm_measures

class EvalGDIScontext:
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config
        self.n_bins = 10
        self.bootstrap_metrics = True
        self.ntimes_bootstrap = 100
        self.ratiosubset = 0.25

        self.names_context_mask = ['RM_levels', 'LU_types', 'SZN_periods']
        self.merged_res = {context_obj: {} for context_obj in self.names_context_mask}
        self.ratio_locations = {context_obj: {} for context_obj in self.names_context_mask}

    def _metrics_(self, name, labels, preds):
        """Compute the set of metrics considered
        """
        metrics = {}
        metrics['AUROC'] = roc_auc_score(labels, preds) 
        metrics['AUPRC'] = average_precision_score(labels, preds)
        metrics['R'] = np.corrcoef(labels, preds)[0, 1]
        
        if name != 'model': 
                #print('--Converting indices values into scores E[0, 1] for ECE and PSNR')
                preds = 1 / (1 + np.exp(-preds))
        metrics['ECE'] = binary_calibration_error(torch.from_numpy(preds), 
                                       torch.from_numpy(labels.astype(int)), 
                                       n_bins = self.n_bins, norm = 'l1').numpy()
        metrics['PSNR'] = 10 * np.log10((1 ** 2) / np.mean((labels - preds)**2)) # data_range = 1. 
        
        return metrics
       
    def _compute_metrics_(self, names, preds, labels):
        """
        """
        metrics_all_names = {}
        for i, name in enumerate(names):
            print('\n-Classification metrics', name)
            
            # Predefine the results
            metrics_all_names[name] = {} 
            metric_names = ['AUROC', 'AUPRC', 'R', 'ECE', 'PSNR'] 
            metrics = {} 
            confidence_intervals = {} 
            stds = {} 
            
            # Perform boostraping of the test set
            if self.bootstrap_metrics:

                print('\n-Bootstrap results', name)
                bootstrap_res = {metric_name: [] for metric_name in metric_names}
                lensubset = int(self.ratiosubset * len(labels))
                n = 0
                while n < self.ntimes_bootstrap:
                    
                    # Split test samples with resampling. 
                    npartition_labels, npartition_preds = resample(labels, preds[i], 
                                                                   n_samples = lensubset, replace = True) 
                
                    # Take a subset that contains True labels
                    if npartition_labels.any():

                        # Compute the metrics for the current partition and store 
                        npartition_metrics = self._metrics_(name, npartition_labels, npartition_preds)
                        for metric_name in metric_names:
                            bootstrap_res[metric_name].append(npartition_metrics[metric_name])

                        # Increase the counter
                        n += 1 
                
                # Compute the average results and the confidence intervals
                for metric_name in metric_names:
                    
                    metrics[metric_name] = bootstrap_res[metric_name]
                    #confidence_intervals[metric_name] = (np.percentile(bootstrap_res[metric_name], 2.5), # "Lower % CI"
                    #                                     np.percentile(bootstrap_res[metric_name], 97.5)) # "Upper % CI"
                    #stds[metric_name] = np.std(bootstrap_res[metric_name])
                
            else:
                all_metrics = self._metrics_(name, labels, preds[i])
                for metric_name in metric_names:
                    metrics[metric_name] = [all_metrics[metric_name]]   
            
            # Display
            print('--Metrics Values', metrics)
            print('--Confidence Intervals', confidence_intervals)
            print('--Stds', stds)
            
            # Save
            metrics_all_names[name] = metrics
            
        return metrics_all_names
    
    def _get_data_on_locations(self, names, data, context_mask = 1):
        """
        """
        # Extract model data
        y_hat = data[0]['y_hat']
        labels = data[0]['labels']
        y_hat_mask = (data[0]['masks'] == 1)
        
        # If we have indices:
        if len(names) == 2:
            
            # Extract index data
            index = data[1][names[1]]
            
            # Get the index mask
            index_mask = self._define_index_mask_(names[1], data[1])
        
            # >>> Create mask combining indexes and model no-nans masks
            # Print the number of event pixels lost due to the index mask
            # (nansum to allow the count when there are nans in the labels. This is 
            # when the oject of reference for the interpolation is the indices).
            # Otherwise there are no nans in the prediction. 
            percent_pixels_after_mask = np.nansum(labels * index_mask) / np.nansum(labels) * 100
            print('-% of event pixels retained after the index mask:', percent_pixels_after_mask)
            print('-% of event pixels lost after the index mask:', 100 - percent_pixels_after_mask)
            assert percent_pixels_after_mask != 0, 'No remaining drought pixels'
            
            # Locations: 
            # avoid the nans from the interpolated indices, 
            # avoid the nans from the input variables and 
            # avoid the nans from the interpolated model.
            locations = index_mask * y_hat_mask * context_mask #* np.logical_not(np.isnan(y_hat))
            
            # Filter for locations
            preds = [y_hat[locations], index[locations]]
            labels = labels[locations]
            
        else: 
            
            # Locations
            locations = y_hat_mask * context_mask
            
            # Filter for locations
            preds = [y_hat[locations]]
            labels = labels[locations]
        
        ratio_locations = np.sum(labels) / np.sum(locations)
        print(f'-Number of locations: {np.sum(locations)}')
        print(f'-Number of droughts: {int(np.sum(labels))}')
        print(f'-Ratio of droughts vs total: {ratio_locations}')

        return preds, labels, ratio_locations 
        
    def _define_index_mask_(self, drought_index_name, indices):
        """
        """
        # >>> Get the mask for the index (common for all indices or individual for each)
        if self.experiment_config['Arguments']['commonMask']:
            
            print('-Setting a common mask for all indices')
            index_mask = []
            for dindex in list(indices.keys()):
                index_mask.append(np.logical_not(np.isnan(indices[dindex])))
            index_mask = (np.prod(np.array(index_mask), axis = 0) == 1) # important to have a bolean mask
            
        else:
            index_mask = np.logical_not(np.isnan(indices[drought_index_name]))
        
        return index_mask
            
    def _evaluation_wrt_context(self, names, data, savefigures = True): 
        """
        """
        print(f'\nEvaluating {names} over context: {self.names_context_mask}')
        
        for context_obj in self.names_context_mask:
            
            context_variable = data[0][context_obj]
            context_classes = np.unique(context_variable)
            context_classes = context_classes[np.isfinite(context_classes)]
            for context_class in context_classes:
                
                # Get elemets on the evaluation locations
                context_mask = (context_variable == context_class)
                
                if savefigures: 
                    plt.figure()
                    plt.imshow(((context_variable == context_class) * data[0]['labels']).sum(axis = 0), cmap = 'viridis'), plt.colorbar()
                    savename = f'{context_obj}_class_{context_class}'
                    plt.title(savename)
                    print_format = 'png' #self.experiment_config['Arguments']['print_format']
                    plt.savefig(self.experiment_config['run_path'] + f'/{savename}.{print_format}')
                    plt.show()
                    plt.close()
                
                print(f'\nEvaluating {names} wrt {context_obj} type {context_class}')
                preds, labels, ratio_locations = self._get_data_on_locations(names, data, context_mask = context_mask)

                tmp = {name: ratio_locations for name in names}
                try: 
                    self.ratio_locations[context_obj][f'C{int(context_class)}'] = {**self.ratio_locations[context_obj][f'C{int(context_class)}'], **tmp}
                except: 
                    self.ratio_locations[context_obj][f'C{int(context_class)}'] = tmp

                # >>> Compute evaluation metrices
                if labels.any() and not labels.all(): 
                    
                    # Add to global variable of results
                    tmp = self._compute_metrics_(names, preds, labels)
                    
                    try: 
                        self.merged_res[context_obj][f'C{int(context_class)}'] = {**self.merged_res[context_obj][f'C{int(context_class)}'], **tmp}
                    except: 
                        self.merged_res[context_obj][f'C{int(context_class)}'] = tmp
                else:
                    print('Locations not valid')
     
    def _save_merged_res(self, model_names):

        with open(self.experiment_config['run_path'] + f'/merged_res_{model_names}.pkl', 'wb') as f: 
            pickle.dump(self.merged_res, f) 

        print('ratio_locations', self.ratio_locations)
        with open(self.experiment_config['run_path'] + f'/ratio_locations_{model_names}.pkl', 'wb') as f: 
            pickle.dump(self.ratio_locations, f) 
    
    def _plot_groupbarplot(self, model_names, res_wrt_context, normalize = False): 
                
        # Set style and define custom color palette
        sns.set_style("whitegrid", {'axes.grid' : False})
        custom_palette = {'SPEI1': '#7DC68B', 'SPEI12': '#B0804C', 
                          'CDI': '#5A96C6', 'EnsembleSMA': '#A9A9A9', 
                          'alpha': '#9D7CDE', 'omega': '#FF7C4C'}
        
        for context_obj in self.names_context_mask:
        
            # Concatenated. Class | Color 
            nexp = self.ntimes_bootstrap if self.bootstrap_metrics else 1
            context_classes = list(res_wrt_context[context_obj])
            classes_refs = [context_class for context_class in context_classes for model_name in model_names for i in range(nexp)]
            hue_refs = [model_name for context_class in context_classes for model_name in model_names for i in range(nexp)] 
            
            # Loop over each metric
            for metric_name in ['AUROC', 'AUPRC']:
                
                # Get values for current metric 
                values = sum([res_wrt_context[context_obj][context_class][model_name][metric_name]
                              for context_class in context_classes for model_name in model_names], []) 
                
                if normalize: 
                    values = (np.array(values) - np.min(values)) / (np.max(values) - np.min(values))
                
                # Create DataFrame
                data = pd.DataFrame({'classes_refs': classes_refs, 'values': values, 'hue_refs': hue_refs})
                
                # Create the plot
                plt.figure(figsize = (14, 8))
                ax = sns.catplot(data = data, x = 'classes_refs', y = 'values', hue = 'hue_refs', 
                                 palette = custom_palette, kind = "bar", legend = False, 
                                 height = 7, width = 0.9, dodge = True, aspect = 1.5,
                                 estimator = 'mean', errorbar = 'sd')
        
                # Labels, grid, ticks and the legend
                plt.xlabel('', fontsize = 26, labelpad = 10)
                plt.ylabel(metric_name, fontsize = 26, labelpad = 10)
                plt.xticks(fontsize = 26)
                plt.yticks(fontsize = 26)
                ax.despine(left = True, bottom = False)  # Hide the top and right spines
                ax.set(ylim = (0, data['values'].max() * 1.1))
                
                # Save
                plt.title(f'boxplot_{model_names}_wrt_{context_obj}_{metric_name}')
                plt.tight_layout()
                savename = f'boxplot_{model_names}_wrt_{context_obj}_{metric_name}'
                print_format = self.experiment_config['Arguments']['print_format']
                plt.savefig(self.experiment_config['run_path'] + f'/{savename}.{print_format}', 
                            bbox_inches = 'tight', pad_inches = 0)
                plt.show()
                plt.close()
        
        
        
     
