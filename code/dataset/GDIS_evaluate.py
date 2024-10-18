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

# PICKLE
import pickle

# TORCH
import torch
from torchmetrics.functional.classification import binary_calibration_error

# UTILS
from utils import cm_measures

class EvalGDIS:
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config
        self.n_bins = 10
        self.model_name = self.experiment_config['Arguments']['model_name'].split('_')[0]
        
        self.ind_names = []
        self.loc_indices_labels = []
        self.loc_indices_preds_model = []
        self.loc_indices_preds_indices = []
        self.loc_model_preds_model = 0
        self.loc_model_labels = 0
        
    def _save_prediccions(self):
        """
        """
        self.predictions_on_locations = {}   
        self.predictions_on_locations['names_models'] = [self.model_name]
        self.predictions_on_locations['loc_model_preds_model_' + self.model_name] = self.loc_model_preds_model 
        self.predictions_on_locations['loc_model_labels_' + self.model_name] = self.loc_model_labels

        self.predictions_on_locations['ind_names'] = self.ind_names
        for i, ind_name in enumerate(self.ind_names):

            tmp_dict = {}
            tmp_dict['loc_indices_preds_model'] = self.loc_indices_preds_model[i]
            tmp_dict['loc_indices_preds_indices'] = self.loc_indices_preds_indices[i]
            tmp_dict['loc_indices_labels'] = self.loc_indices_labels[i]
            self.predictions_on_locations[ind_name] = tmp_dict
        
        # Save outputs to perform evaluation
        with open(self.experiment_config['run_path'] + 
                  f"/predictions_on_locations_{self.model_name}.pkl", "wb") as f: 
            pickle.dump(self.predictions_on_locations, f)
    
    def _metrics_(self, name, labels, preds):
        """Compute the set of metrics considered
        """
        metrics = {}
        metrics['AUROC'] = roc_auc_score(labels, preds) 
        metrics['AP'] = average_precision_score(labels, preds)
        metrics['R'] = np.corrcoef(labels, preds)[0, 1]
        
        if name != 'model': 
                #print('--Converting indices values into scores E[0, 1] for ECE and PSNR')
                preds = 1 / (1 + np.exp(-preds))
        metrics['ECE'] = binary_calibration_error(torch.from_numpy(preds), 
                                       torch.from_numpy(labels.astype(int)), 
                                       n_bins = self.n_bins, norm = 'l1').numpy()
        metrics['PSNR'] = 10 * np.log10((1 ** 2) / np.mean((labels - preds)**2)) # data_range = 1. 
        
        return metrics
       
    def _compute_metrics_(self, names, preds, labels,
                          bootstrap_metrics = True, 
                          ntimes_bootstrap = 100,
                          ratiosubset = 0.25):
        """
        """
        for i, name in enumerate(names):
            print('\n-Classification metrics', name)
            
            # Predefine the results 
            metric_names = ['AUROC', 'AP', 'R', 'ECE', 'PSNR'] 
            metrics = {metric_name: np.nan for metric_name in metric_names}
            confidence_intervals = {metric_name: np.nan for metric_name in metric_names}
            stds = {metric_name: np.nan for metric_name in metric_names}
            
            # Perform boostraping of the test set
            if bootstrap_metrics:

                print('\n-Bootstrap results', name)
                bootstrap_res = {metric_name: [] for metric_name in metric_names}
                lensubset = int(ratiosubset * len(labels))
                n = 0
                while n < ntimes_bootstrap:
                    
                    # Split test samples with resampling. 
                    npartition_labels, npartition_preds = resample(labels, preds[i], #stratify = preds[i],
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
                    
                    metrics[metric_name] = np.mean(bootstrap_res[metric_name])
                    confidence_intervals[metric_name] = (np.percentile(bootstrap_res[metric_name], 2.5), # "Lower % CI"
                                                         np.percentile(bootstrap_res[metric_name], 97.5)) # "Upper % CI"
                    stds[metric_name] = np.std(bootstrap_res[metric_name])
                
            else:
                metrics = self._metrics_(name, labels, preds[i])
        
            # Display
            print('Metrics Values', metrics)
            print('Confidence Intervals', confidence_intervals)
            print('Stds', stds)
        
    def _plot_classification_curves(self, names, preds, labels):
        """
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].plot([0, 1], [0, 1], "k:", label = "Chance")
        axs[1].plot([0, 1], [1, 0], "k:", label = "Chance")
        for i, name in enumerate(names):
            
            # AUROC
            roc_display = RocCurveDisplay.from_predictions(
                            labels, preds[i],
                            ax = axs[0], name = name,
                            color = ['b', 'r'][i],
                            linewidth = 1.0)
                
            # AVERAGE PRECISION 
            ap_display = PrecisionRecallDisplay.from_predictions(
                            labels, preds[i],
                            ax = axs[1], name = name, 
                            color = ['b', 'r'][i],
                            linewidth = 1.0)
            
        axs[0].set_title('ROC curve')
        axs[1].set_title('Precision-Recall curve')
        plt.tight_layout()
        plt.savefig(self.experiment_config['run_path'] + '/classification_curves' + '_'.join(names) + '.png')
        plt.show()
        plt.close()
        
    def _plot_AUROCcurves(self):
        """
        """
        fig = plt.figure(figsize=(10, 10))
        plt.plot([0, 1], [0, 1], "k:", label = "Chance", linewidth = 6)
        
        # Alpha curve 
        if 'alpha' in self.predictions_on_locations['names_models']:
            
            fpr, tpr, _ = roc_curve(self.predictions_on_locations['loc_model_labels_alpha'],
                                    self.predictions_on_locations['loc_model_preds_model_alpha'])
            plt.plot(fpr, tpr, label = 'NL', color = "#9D7CDE", linestyle = "-", linewidth = 6)
        
        # Omega curve
        if 'omega' in self.predictions_on_locations['names_models']:
            
            fpr, tpr, _ = roc_curve(self.predictions_on_locations['loc_model_labels_omega'],
                                    self.predictions_on_locations['loc_model_preds_model_omega'])
            plt.plot(fpr, tpr, label = 'SPLSC', color = '#FF7C4C', linestyle = "-", linewidth = 6)
    
        # Indices curves
        colors = ["#7DC68B", "#B0804C", "#5A96C6", "#A9A9A9"]
        for i, ind_name in enumerate(self.predictions_on_locations['ind_names']):
            fpr, tpr, _ = roc_curve(self.predictions_on_locations[ind_name]['loc_indices_labels'], 
                                    self.predictions_on_locations[ind_name]['loc_indices_preds_indices'])
            plt.plot(fpr, tpr, label = ind_name, color = colors[i], linestyle = "-", linewidth = 6)
        
        plt.xlabel('FPR', fontsize = 22)
        plt.ylabel('TPR', fontsize = 22)
        plt.xticks(fontsize = 22)
        plt.yticks(fontsize = 22)
        plt.legend(fontsize = 20, loc='lower right')
        plt.savefig((self.experiment_config['run_path'] + f"/AUROC_curves.{self.experiment_config['Arguments']['print_format']}"), 
                    format = self.experiment_config['Arguments']['print_format'], bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        plt.close(fig = fig)
        
    def _plot_reliability_diagrams(self, names, preds, labels):
        """It plots the true frequency of the positive label 
        against its predicted probability, for binned predictions.
        The x axis represents the mean predicted probability in each bin.
        The y axis is the fraction of positives (in each bin)
        It is recommended that a proper probability is used for pred
        Intuitively, the calibration plot tells that:
        when the average predictive probability is XX, 
        about how much percent of the predictions are positive
        https://towardsdatascience.com/introduction-to-reliability-diagrams-for-probability-calibration-ed785b3f5d44
        """
        plt.figure()
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
    
        ax1.plot([0, 1], [0, 1], "k:", label = "Perfectly calibrated")
        for i, name in enumerate(names):
            
            # Adjust
            if name != 'model': 
                print('--Converting indices values into scores E[0, 1]')
                tmp_preds = 1 / (1 + np.exp(-preds[i]))
            
            else:
                tmp_preds = preds[i]
            
            # compute the fraction of positives and mean predicted value
            fp, mean_pv = calibration_curve(labels, tmp_preds, n_bins = self.n_bins)
            
            # Plot curve and histogram
            ax1.plot(mean_pv, fp, "s-", label = name)
            ax2.hist(tmp_preds, range = (0, 1), bins = self.n_bins, label = name, histtype = "step")
            
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc = "upper left")
        ax1.set_title('Calibration plots (reliability curve)')
        
        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc = "upper right")
        
        plt.tight_layout()
        plt.savefig(self.experiment_config['run_path'] + '/reliability_diagrams_' + '_'.join(names) + '.png')
        plt.show()
        plt.close()
    
    def _get_data_on_locations(self, names, data):
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
            locations = index_mask * y_hat_mask #* np.logical_not(np.isnan(y_hat))
            
            # Filter for locations
            preds = [y_hat[locations], index[locations]]
            labels = labels[locations]
            
        else: 
            
            # Locations
            locations = y_hat_mask
            
            # Filter for locations
            preds = [y_hat[locations]]
            labels = labels[locations]
            
        return preds, labels
        
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
    
    def _evaluation(self, names, data, 
                    classification_curves = True, 
                    reliability_diagrams = True):
        """
        """
        print('\nEvaluating', names)
        
        # Get elemets on the evaluation locations
        preds, labels = self._get_data_on_locations(names, data)
        
        # >>> Compute evaluation metrices
        if labels.any() and not labels.all(): 
            self._compute_metrics_(names, preds, labels, bootstrap_metrics = False)
            #self._compute_metrics_(names, preds, labels, bootstrap_metrics = True)
        
        else:
            print('Locations not valid')
        
        # Store values for later use 
        if len(names) > 1:
            
            self.ind_names.append(names[1])
            self.loc_indices_labels.append(labels)
            self.loc_indices_preds_model.append(preds[0])
            self.loc_indices_preds_indices.append(preds[1])
        
        else:
        
            self.loc_model_preds_model = preds[0]
            self.loc_model_labels = labels

        # Individual pair of plots
        if classification_curves: 
            self._plot_classification_curves(names, preds, labels)
             
        if reliability_diagrams: 
            self._plot_reliability_diagrams(names, preds, labels)
