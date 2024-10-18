#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY
import numpy as np

# MATPLOTLIB
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# SKLEARN 
from sklearn.metrics import roc_curve

# CARTOPY
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator

# PICKLE
import pickle

# PANDAS
import pandas as pd

# UTILS
from utils import normalize_timesteps, normalize_events, geoplot2d, define_nonoverlaping_3Dforms

class CompGDIS:
    def __init__(self, dataset_config, experiment_config):
        """
        """
        self.experiment_config = experiment_config
        
        # Modifications to the bounding forms 
        self.border3Dforms = eval(dataset_config['GDIS']['border3Dforms'])
        self.plot_3Dforms = dataset_config['GDIS']['plot_3Dforms']
        self.margins3Dforms = dataset_config['GDIS']['margins3Dforms']

        # Correction for the model's lost border and resolution correspondence
        size_in = list(eval(dataset_config['GDIS']['input_size']))
        size_out = list(eval(dataset_config['GDIS']['output_size']))
        resolution_adjust = int((size_in[0] - size_out[0])/2) * dataset_config['GDIS']['resolution']
        self.vismargin = dataset_config['GDIS']['vismargin']
        resolution_adjustvismargin = self.vismargin * dataset_config['GDIS']['resolution']      
        self.im_extent = (eval(dataset_config['GDIS']['lon_slice'])[0] + resolution_adjust, 
                          eval(dataset_config['GDIS']['lon_slice'])[1] - resolution_adjustvismargin,
                          eval(dataset_config['GDIS']['lat_slice'])[1] + resolution_adjustvismargin,
                          eval(dataset_config['GDIS']['lat_slice'])[0] - resolution_adjust) 
        
        central_longitude = (self.im_extent[1] - abs(self.im_extent[0]))/2
        central_latitude = (self.im_extent[3] - abs(self.im_extent[2]))/2
        self.central_lat_lon = {'central_longitude': central_longitude, 'central_latitude': central_latitude}
                                      
    def _compare_models_(self, model_alpha, model_omega, time_ref, normalize_maps = True, print_format = 'png'):
        """
        """
        print('\nModel comparison...')
        
        # Get the outputs of the models
        # (and turn nan the locations where the input was zero)
        maps_alpha = np.where(model_alpha['masks'] == 0, np.nan, model_alpha['y_hat'])
        maps_omega = np.where(model_omega['masks'] == 0, np.nan, model_omega['y_hat'])
        
        # Get the labels and map of events
        maps_labels = model_alpha['labels']
        maps_events = model_alpha['map_events']
        
        # Get events ids
        mask_event_ids = np.unique(maps_events)
        
        # Predefine the probabilities signals (-999 means all events)
        probs_signals_alpha = {element:[] for element in np.append(mask_event_ids, -999)} 
        probs_signals_omega = {element:[] for element in np.append(mask_event_ids, -999)} 
        probs_signals_labels = {element:[] for element in np.append(mask_event_ids, -999)} 
        
        ##################################################################
        # Create a copy of the maps. 
        # If normalizing, the probabilities should not be changed, that's why we copy the variables for plotting
        plot_maps_alpha = maps_alpha.copy()
        plot_maps_omega = maps_omega.copy()
        if normalize_maps:
            
            print('--Normalizing output maps')
            plot_maps_alpha = (maps_alpha - np.nanmin(model_alpha['y_hat'])) / (np.nanmax(model_alpha['y_hat']) - np.nanmin(model_alpha['y_hat']))
            plot_maps_omega = (maps_omega - np.nanmin(model_omega['y_hat'])) / (np.nanmax(model_omega['y_hat']) - np.nanmin(model_omega['y_hat']))
        
        print('-Plot maps')    
        # Plot the aggregated map (mean predictions over time)
        aggregated_alpha = np.nanmean(plot_maps_alpha, axis = 0)
        aggregated_alpha = (aggregated_alpha - np.nanmin(aggregated_alpha))/(np.nanmax(aggregated_alpha) - np.nanmin(aggregated_alpha))
        aggregated_alpha = np.where(np.prod(model_alpha['masks'], axis = 0) == 0, np.nan, aggregated_alpha)
        
        aggregated_omega = np.nanmean(plot_maps_omega, axis = 0)
        aggregated_omega = (aggregated_omega - np.nanmin(aggregated_omega))/(np.nanmax(aggregated_omega) - np.nanmin(aggregated_omega))
        aggregated_omega = np.where(np.prod(model_omega['masks'], axis = 0) == 0, np.nan, aggregated_omega)
        
        self._map_differences_one_plot_(aggregated_alpha[:-self.vismargin, :-self.vismargin] - aggregated_omega[:-self.vismargin, :-self.vismargin], 
                                        (np.sum(maps_labels, axis = 0) != 0)[:-self.vismargin, :-self.vismargin], 
                                        title = 'All_dif_one_plot_BaselinevsSPX', print_format = print_format)
        
        self._map_sidebyside_(aggregated_alpha[:-self.vismargin, :-self.vismargin], 
                              aggregated_omega[:-self.vismargin, :-self.vismargin], 
                              (np.sum(maps_labels, axis = 0) != 0)[:-self.vismargin, :-self.vismargin], 
                              title = 'All_diff', print_format = print_format) 
        
        # >>> Plot all prediction BOXES      
        expanded_forms3D = define_nonoverlaping_3Dforms(maps_events[:,:,:-self.vismargin, :-self.vismargin], 
                                                        model_alpha['labels'][:,:-self.vismargin, :-self.vismargin],
                                                        border3Dforms = self.border3Dforms,
                                                        margin = self.margins3Dforms,  
                                                        plot_3Dforms = self.plot_3Dforms,
                                                        save_path = self.experiment_config['run_path'])
        
        ### ALPHA
        # Boxes maps
        bounded_maps_alpha = np.where(expanded_forms3D == 0, np.nan, plot_maps_alpha[:, :-self.vismargin, :-self.vismargin])
        bounded_maps_alpha = np.nanmean(bounded_maps_alpha, axis = 0)
        
        # Combined and final
        combined_maps_alpha = np.where((np.prod(model_alpha['masks'], axis = 0) == 0)[:-self.vismargin, :-self.vismargin], 
                                       np.nan, bounded_maps_alpha)
        
        ### OMEGA
        # Boxes maps
        bounded_maps_omega = np.where(expanded_forms3D == 0, np.nan, plot_maps_omega[:, :-self.vismargin, :-self.vismargin])
        bounded_maps_omega = np.nanmean(bounded_maps_omega, axis = 0)
        
        # Combined and final
        combined_maps_omega = np.where((np.prod(model_alpha['masks'], axis = 0) == 0)[:-self.vismargin, :-self.vismargin], 
                                       np.nan, bounded_maps_omega)

        ### ALPHA - OMEGA. Range: [-1, 1]
        diff_combined = combined_maps_alpha - combined_maps_omega
        diff_combined = (2 * (diff_combined - np.nanmin(diff_combined)) / (np.nanmax(diff_combined) - np.nanmin(diff_combined))) - 1

        self._map_sidebyside_(combined_maps_alpha, 
                              combined_maps_omega, 
                              (np.sum(maps_labels, axis = 0) != 0)[:-self.vismargin, :-self.vismargin], 
                              (np.sum(expanded_forms3D, axis = 0) != 0),
                              title = 'All_diff_bounded', print_format = print_format)
        
        self._map_differences_one_plot_(diff_combined, 
                                        (np.sum(maps_labels, axis = 0) != 0)[:-self.vismargin, :-self.vismargin], 
                                        (np.sum(expanded_forms3D, axis = 0) != 0),
                                        title = 'All_dif_one_plot_BaselinevsSPX_bounded', print_format = print_format)
        
        # Plot the aggregated map (mean predictions over time). Percentile version
        print('len(time_ref)', len(time_ref))
        print('--------------STATS------------') 
        dummy = np.where(model_omega['masks'] == 0, np.nan, maps_labels)
        print('n_drought', np.nansum(dummy == 1))
        print('n_nodrought', np.nansum(dummy == 0))
        print('ratio', np.nanmean(maps_labels))

        ##################################################################
        # Loop over all time instances
        for i, time in enumerate(time_ref):
            print('-Sample number: ', i)
            
            # Plot the timestep maps side by side
            self._map_sidebyside_(plot_maps_alpha[i][:-self.vismargin, :-self.vismargin], 
                                  plot_maps_omega[i][:-self.vismargin, :-self.vismargin], 
                                  maps_labels[i][:-self.vismargin, :-self.vismargin], 
                                  title = np.datetime_as_string(time, unit='D'), print_format = print_format) 
        
            # Probs (signals)
            # all elements
            if (maps_events[i] != 0).any():

                probs_signals_alpha[-999].append(maps_alpha[i, (maps_events[i] != 0).any(axis = 0)])
                probs_signals_omega[-999].append(maps_omega[i, (maps_events[i] != 0).any(axis = 0)])
                probs_signals_labels[-999].append(maps_labels[i, (maps_events[i] != 0).any(axis = 0)])
            
            else:

                probs_signals_alpha[-999].append([0])
                probs_signals_omega[-999].append([0])
                probs_signals_labels[-999].append([0])
            
            # individual elements
            for event_id in mask_event_ids:
                #print(event_id, mask_event_ids)
                if (maps_events[i] == event_id).any():
                    
                    probs_signals_alpha[event_id].append(maps_alpha[i, (maps_events[i] == event_id).any(axis = 0)])
                    probs_signals_omega[event_id].append(maps_omega[i, (maps_events[i] == event_id).any(axis = 0)])
                    probs_signals_labels[event_id].append(maps_labels[i, (maps_events[i] == event_id).any(axis = 0)])
                
                else: 
                
                    probs_signals_alpha[event_id].append([0])
                    probs_signals_omega[event_id].append([0])
                    probs_signals_labels[event_id].append([0])
        
        # Plot the signals
        print('-Plot signals')
        probs_signals_alpha = self._preprocess_signals_(probs_signals_alpha, normalized_events = True, labels_as_True = False,
                                                        smooth = True, smooth_type='ewm', alpha = 0.25)
        probs_signals_omega = self._preprocess_signals_(probs_signals_omega, normalized_events = True, labels_as_True = False,
                                                        smooth = True, smooth_type='ewm', alpha = 0.25)
        probs_signals_labels = self._preprocess_signals_(probs_signals_labels, normalized_events = True, labels_as_True = True)

        self._sig_differences_one_plot_(time_ref, probs_signals_alpha, probs_signals_omega, probs_signals_labels,
                              set_legend = False, print_format = print_format)
        
        self._sig_sidebyside_(time_ref, probs_signals_alpha, probs_signals_omega, probs_signals_labels,
                              set_legend = False, print_format = print_format)

        # Plot the AUROC curves        
        print('-AUROC curves')
        try:
            
            self._load_preds_on_locations()
            self._plot_AUROCcurves()
        
        except: 
            print('No data to plot the AUROC curves')
    
    def _load_preds_on_locations(self):
        """
        """
        with open(self.experiment_config['run_path'] + '/predictions_on_locations_alpha.pkl', "rb") as f:
            self.predictions_on_locations = pickle.load(f)

        with open(self.experiment_config['run_path'] + '/predictions_on_locations_omega.pkl', "rb") as f:
            predictions_on_locations_omega = pickle.load(f)

        # Change the names and add the omega 
        self.predictions_on_locations['names_models'] = ['alpha', 'omega']
        self.predictions_on_locations['loc_model_labels_omega'] = predictions_on_locations_omega['loc_model_labels_omega']
        self.predictions_on_locations['loc_model_preds_model_omega'] = predictions_on_locations_omega['loc_model_preds_model_omega']
    
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
        plt.savefig((self.experiment_config['run_path'] + f"/Comparison_AUROC_curves.{self.experiment_config['Arguments']['print_format']}"), 
                    format = self.experiment_config['Arguments']['print_format'], bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        plt.close(fig = fig)
    
    def _map_differences_one_plot_(self, map_difference, map_labels, bbox_contour = np.zeros(0), title = '', print_format = 'png'):
        """
        """
        fig = plt.figure(figsize = (10, 14)) 
        ax = plt.subplot(1, 1, 1, projection = ccrs.LambertConformal(**self.central_lat_lon)) 
        im = geoplot2d(ax, map_difference, 
                       self.im_extent, 
                       im_contour = map_labels, 
                       bbox_contour = bbox_contour,
                       colors_ref = 'PuOr', set_colorbar = False)
        im.set_clim(-1, 1)
        
        #fig.colorbar(im, ax=ax, orientation = 'horizontal')
        plt.savefig(self.experiment_config['run_path'] + '/' + title + f'.{print_format}', 
                    format = print_format, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        plt.close(fig = fig)
    
    def _map_sidebyside_(self, map_alpha, map_omega, map_labels, bbox_contour = np.zeros(0), title = '', print_format = 'png'):
        """
        """
        fig = plt.figure(figsize = (10, 14)) 
        ax1 = plt.subplot(3, 1, 1, projection = ccrs.LambertConformal(**self.central_lat_lon))
        ax2 = plt.subplot(3, 1, 2, projection = ccrs.LambertConformal(**self.central_lat_lon))
        ax3 = plt.subplot(3, 1, 3, projection = ccrs.LambertConformal(**self.central_lat_lon))

        im1 = geoplot2d(ax1, map_alpha, self.im_extent, im_contour = map_labels, 
                        bbox_contour = bbox_contour, set_colorbar = False)
        im2 = geoplot2d(ax2, map_omega, self.im_extent, im_contour = map_labels, 
                        bbox_contour = bbox_contour, set_colorbar = False)
        im3 = geoplot2d(ax3, map_alpha - map_omega, self.im_extent, im_contour = map_labels, 
                        bbox_contour = bbox_contour, colors_ref = 'viridis', set_colorbar = False)
        """
        im1.set_clim(0, 1)
        im2.set_clim(0, 1)
        im3.set_clim(-1, 1)
        """
        fig.colorbar(im1, ax=ax1, orientation = 'vertical')
        fig.colorbar(im2, ax=ax2, orientation = 'vertical')
        fig.colorbar(im3, ax=ax3, orientation = 'vertical')
        
        ax1.title.set_text('Alpha')
        ax2.title.set_text('Omega')
        ax3.title.set_text('Alpha - Omega')
        
        fig.tight_layout()
        plt.savefig(self.experiment_config['run_path'] + '/' + title + f'.{print_format}', 
                    format = print_format, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        plt.close(fig = fig)
        
    def _preprocess_signals_(self, signals, 
                             metric = 'mean', 
                             normalized_timesteps = False, 
                             normalized_events = True,
                             labels_as_True = True, 
                             smooth = False, smooth_type='ewm', alpha = np.nan):
        """
        """
        # Correction for the number of elements that define each timestep 
        if normalized_timesteps: 
            # Normalize each timestep correcting for the number of pixels
            # involved in that timestep and the total of pixels involved
            print('--Normalizing timesteps')
            signals = normalize_timesteps(signals)
        
        # Get the probabilites according to the metric
        signals_metric = {key: [] for key in signals.keys()}
        for key in signals.keys():
            for timestep in range(len(signals[key])):
                
                if metric == 'mean':
                    signals_metric[key].append(np.nanmean(signals[key][timestep]))
        
                elif metric == 'median':
                    signals_metric[key].append(np.nanmedian(signals[key][timestep]))
            
        # Normalize according to maximum value in all events
        if normalized_events: 
            
            print('--Normalizing events')
            signals_metric = normalize_events(signals_metric)
            
        if labels_as_True:
            
            # Sets the labels to 1 if there is at least one pixel true. Ignores normalization of labels and the mean
            print('--Labels for timestep set to one if one True pixel exists')
            tmp_signals_metric = {key: (signals_metric[key] > 0) for key in signals_metric.keys()}
            signals_metric = tmp_signals_metric

        if smooth: 
            if smooth_type == 'ewm':
                for ikey in np.array(list(signals_metric.keys())):
                    df = pd.DataFrame(data = np.array(signals_metric[ikey]))
                    df = df.ewm(alpha = alpha).mean()
                    signals_metric[ikey] = df.to_numpy().ravel()
            
        return signals_metric
           
    def _sig_differences_one_plot_(self, time_ref, probs_signals_alpha, probs_signals_omega, probs_signals_labels,
                              set_legend = False, yscale = 'linear', print_format = 'png'):
    
        # Elements to plot and background
        elements = np.array(list(probs_signals_alpha.keys()))
        elements = elements[elements != 0]
        background = 0
        
        # Plot signal for each element
        for event in elements:

            fig, ax = plt.subplots(1,1, figsize=(10, 8))
            plt.plot(time_ref, probs_signals_labels[event], color='#8D8D8D', linestyle='-', label='Drought GT', linewidth=6) 
            plt.plot(time_ref, probs_signals_alpha[event], color='#9D7CDE', linestyle='-', label='Baseline', linewidth=6)
            plt.plot(time_ref, probs_signals_omega[event], color='#FF7C4C', linestyle='-', label='SPX', linewidth=6)
            #plt.grid(True, axis = 'y', linestyle='--', alpha = 0.5)
            #ax = plt.gca()
            #ax.spines[['top', 'right']].set_visible(False) 
            plt.yscale(yscale)
            plt.xticks(fontsize = 26) #24
            plt.yticks(fontsize = 26)
            plt.ylabel('$\overline{Y}(t)$', fontsize = 26)    
            if set_legend:
                plt.legend(fontsize = 20)
            # Set the x-axis locator to show ticks only for years
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.gcf().autofmt_xdate()
            plt.savefig((self.experiment_config['run_path'] + f'/Signals_differences_' + str(np.round(event)) + f'.{print_format}'), 
                        format = print_format, bbox_inches = 'tight', pad_inches = 0)
            plt.show()
            plt.close(fig = fig)
    
    def _sig_sidebyside_(self, time_ref, probs_signals_alpha, probs_signals_omega, probs_signals_labels, 
                         set_legend = True, yscale = 'linear', print_format = 'png'):
        """
        """
        # Elements to plot and background
        elements = np.array(list(probs_signals_alpha.keys()))
        elements = elements[elements != 0]
        background = 0
        
        # Plot signal for each element
        for event in elements:
            
            fig, axs = plt.subplots(2, 1, figsize = (10, 8))
            axs[0].plot(time_ref, probs_signals_labels[event], c = '#8D8D8D', label = 'Drought GT', linewidth = 4)
            axs[0].plot(time_ref, probs_signals_alpha[event], c = '#1C2D38', label = 'DS_alpha', linewidth = 4, linestyle = 'solid')
            axs[0].plot(time_ref, probs_signals_omega[event], c = '#E04FA6', label = 'DS_omega', linewidth = 4, linestyle = 'solid')
            
            axs[1].plot(time_ref, probs_signals_labels[event], c = '#8D8D8D', label = 'Drought GT', linewidth = 4)
            axs[1].plot(time_ref, probs_signals_alpha[event] - probs_signals_alpha[background], c = 'midnightblue', label = 'signal_lessBGS_alpha', linewidth = 4, linestyle = 'solid') 
            axs[1].plot(time_ref, probs_signals_omega[event] - probs_signals_omega[background], c = 'orchid', label = 'signal_lessBGS_omega', linewidth = 4, linestyle = 'solid') 
            
            axs[0].xaxis.set_tick_params(labelsize = 20)
            axs[0].yaxis.set_tick_params(labelsize = 20)
            axs[0].set_ylabel('$\overline{Y}(t)$', fontsize = 20)    
            axs[0].set_yscale(yscale)   
            
            axs[1].xaxis.set_tick_params(labelsize = 20)
            axs[1].yaxis.set_tick_params(labelsize = 20)
            axs[1].set_ylabel('$\overline{Y}(t)$', fontsize = 20)    
            axs[1].set_yscale(yscale)
              
            if set_legend:
                axs[0].legend(fontsize = 20) 
                axs[1].legend(fontsize = 20)

            plt.gcf().autofmt_xdate()
            plt.savefig(self.experiment_config['run_path'] + '/Signals_' + str(np.round(event)) + f'.{print_format}', 
                        format = print_format, bbox_inches = 'tight', pad_inches = 0)
            plt.show()
            plt.close(fig = fig)
        
