#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY 
import numpy as np

# PYTORCH
import torch

# PICKLE
import pickle

# TIME
import time

# UTILS
from utils import plot_GDIS_cm, plot_GDIS_map, plot_GDIS_signals, crop_variable, get_relevance_map, define_nonoverlaping_3Dforms

def do_inference(model, loader_test, dataset_config, save_path, print_format = 'png'):
    """Function to perform inference and evaluate model 

    :param model: model class (initialized object)
    :type model: pytorch_lightning.LightningModule
    :param loader_test: dataloader 
    :type loader_test: torch.utils.data.DataLoader
    :param dataset_config: dataset configuration
    :type dataset_config: dict
    :param save_path: path to save the results
    :type save_path: str
    :param print_format: how to save images, defaults to 'png'
    :type print_format: str, optional
    """    
    # Set model to evaluate
    model.eval()
    with torch.no_grad():
        
        # >>> Predefine variables
        # Get events ids
        mask_event_ids = loader_test.dataset.events_ids.cpu()
        # setting no drought as an event with id_value= 0.0
        mask_event_ids = np.append(mask_event_ids, 0)
        
        # >>> Save outputs and predicctions for evaluation against indices
        results_maps = {'y_hat':[], 'labels': [], 'masks':[], 'spx_labels': [],
                        'id_events': mask_event_ids, 'map_events': []}
        
        # Define probabilities signals
        probs_signals_yhat = {key:[] for key in np.append(mask_event_ids, -999)}
        probs_signals_labels = {key:[] for key in np.append(mask_event_ids, -999)} 
        
        # To correct for missing positions in the output
        size_in = list(eval(dataset_config['GDIS']['input_size']))
        size_out = list(eval(dataset_config['GDIS']['output_size']))
        lost_border = int((size_in[0] - size_out[0])/2) 
        idt = dataset_config['GDIS']['idt']
        time_border = dataset_config['GDIS']['idt_ref_in'][idt]
        vismargin = dataset_config['GDIS']['vismargin'] # visualization purposes
 
        # Times
        # do take into acount that we start by taking into acount the missed temporal instances
        # and that the end of the sequence is missing some elements 
        times = loader_test.dataset.data.time.values[time_border:][:len(loader_test)]

        # Loop over the dataset
        dataiter = iter(loader_test) # The iter() function creates an object which can be iterated one element at a time.
        for i in np.arange(len(loader_test)):
            print('Sample number: ', i)
            
            # Get samples
            x, masks, labels, map_dominant_context, _, mask_event_locations = next(dataiter) 
            
            # Evaluate 
            start_time = time.time()
            res = model.evaluate_map(x, masks, labels, mode = 'test')
            inference_time_secs_it = round((time.time() - start_time), 2)
            print('inference_time_secs_it:', inference_time_secs_it, 'sec/loop over Europe')
            y_hat = res['output']
            labels = res['labels']
            import sys
            sys.exit()
            # Correct for the empty border that is added in the superior and left side of the inputs
            # during the reconstruction step of the map
            masks = masks[:,:,:, 
                          lost_border:,
                          lost_border:]
            map_dominant_context = map_dominant_context[:,:,:,
                                                        lost_border:,
                                                        lost_border:]
            mask_event_locations = mask_event_locations[:,:,:,
                                                        lost_border:,
                                                        lost_border:]
            
            # Adapt variables to the output size
            masks = crop_variable(dataset_config['GDIS'], masks, temporal_crop = True)
            map_dominant_context = crop_variable(dataset_config['GDIS'], map_dominant_context, temporal_crop = True)
            mask_event_locations = crop_variable(dataset_config['GDIS'], mask_event_locations, temporal_crop = True)
            
            # Converto to numpy, select time frame and remove extra dimensions (batch and variable)
            # we take the first temporal instance
            y_hat = y_hat[0][0][idt].cpu().numpy()
            labels = labels[0][0][idt].cpu().numpy()
            masks = masks[0][0][idt].cpu().numpy()
            mask_event_locations = mask_event_locations[0][:, idt].cpu().numpy()
            map_dominant_context = map_dominant_context[0][0][idt].cpu().numpy()
            
            # Save output to perform evaluation
            results_maps['y_hat'].append(np.copy(y_hat))
            results_maps['labels'].append(np.copy(labels))
            results_maps['masks'].append(np.copy(masks))
            results_maps['map_events'].append(np.copy(mask_event_locations))
            
            # turn nan the output where the input was zero
            y_hat[masks==0] = np.nan
            
            # Probs (signals)
            # all elements
            if (mask_event_locations != 0).any():
                probs_signals_yhat[-999].append(y_hat[(mask_event_locations != 0).any(axis = 0)])
                probs_signals_labels[-999].append(labels[(mask_event_locations != 0).any(axis = 0)])
            
            else:
                probs_signals_yhat[-999].append([0])
                probs_signals_labels[-999].append([0])
            
            # individual elements
            for event_id in mask_event_ids:
                if (mask_event_locations == event_id).any():
                    probs_signals_yhat[event_id].append(y_hat[(mask_event_locations == event_id).any(axis = 0)])
                    probs_signals_labels[event_id].append(labels[(mask_event_locations == event_id).any(axis = 0)])
                else: 
                    probs_signals_yhat[event_id].append([0])
                    probs_signals_labels[event_id].append([0])
        
        # Save outputs to perform evaluation
        with open(save_path + '/evaluate_results.pkl', "wb") as f: 
            pickle.dump(results_maps, f)
        
        # Normalize the maps according to the max prob
        normalized_maps = (np.array(results_maps['y_hat']) - np.nanmin(results_maps['y_hat'])) / (np.nanmax(results_maps['y_hat']) - np.nanmin(results_maps['y_hat']))
        print('Min probability:', np.nanmin(results_maps['y_hat']))
        print('Max probability:', np.nanmax(results_maps['y_hat']))
        
        # >>> Plot all prediction BOXES
        expanded_forms3D = define_nonoverlaping_3Dforms(np.array(results_maps['map_events'])[:,:,:-vismargin, :-vismargin], 
                                                        np.array(results_maps['labels'])[:, :-vismargin, :-vismargin],
                                                        border3Dforms = eval(dataset_config['GDIS']['border3Dforms']), 
                                                        plot_3Dforms = dataset_config['GDIS']['plot_3Dforms'], 
                                                        margin = dataset_config['GDIS']['margins3Dforms'],
                                                        save_path = save_path)

        #Plot label map
        for i in np.arange(len(loader_test)):
            
            #Plot label map
            title = str(times[i])[:10] 
            plot_GDIS_map(title, 
                          normalized_maps[i][:-vismargin, :-vismargin], 
                          results_maps['masks'][i][:-vismargin, :-vismargin], 
                          results_maps['labels'][i][:-vismargin, :-vismargin], 
                          loader_test.dataset, dataset_config, save_path,
                          set_colorbar = False,
                          bbox_contour = expanded_forms3D[i], 
                          drought_id = None, print_format = print_format)
        
        # >>> Signals
        plot_GDIS_signals(times, probs_signals_yhat, probs_signals_labels, save_path, 
                        metric = 'mean', normalized_events = True, labels_as_True = True, 
                        set_legend = False, print_format = print_format)
        
        plot_GDIS_signals(times, probs_signals_yhat, probs_signals_labels, save_path, 
                        metric = 'mean', normalized_events = True, labels_as_True = True, 
                        smooth = True, smooth_type='ewm', alpha = 0.25,
                        set_legend = False, print_format = print_format)
        
        plot_GDIS_signals(times, probs_signals_yhat, probs_signals_labels, save_path, 
                        metric = 'mean', normalized_events = True, labels_as_True = True, 
                        smooth = True, smooth_type='ewm', alpha = 0.25,
                        set_legend = False, print_format = print_format)
        
        # >>> Plot all prediction/label map (MEAN)
        title = 'All_mean_normalized'
        acumulated_yhat = np.mean(results_maps['y_hat'], axis = 0)
        acumulated_yhat = (acumulated_yhat - np.nanmin(acumulated_yhat))/(np.nanmax(acumulated_yhat) - np.nanmin(acumulated_yhat))
        acumulated_masks = np.sum(results_maps['masks'], axis = 0)
        acumulated_labels = (np.sum(results_maps['labels'], axis = 0) != 0)
        plot_GDIS_map(title, 
                      acumulated_yhat[:-vismargin, :-vismargin], 
                      acumulated_masks[:-vismargin, :-vismargin], 
                      acumulated_labels[:-vismargin, :-vismargin], 
                      loader_test.dataset, dataset_config, save_path, 
                      drought_id = None, set_colorbar = False, print_format = print_format)
        
        # >>> Plot all prediction BOXES
        expanded_forms3D = define_nonoverlaping_3Dforms(np.array(results_maps['map_events'])[:,:,:-vismargin, :-vismargin], 
                                                        np.array(results_maps['labels'])[:, :-vismargin, :-vismargin],
                                                        border3Dforms = eval(dataset_config['GDIS']['border3Dforms']), 
                                                        plot_3Dforms = dataset_config['GDIS']['plot_3Dforms'], 
                                                        margin = dataset_config['GDIS']['margins3Dforms'],
                                                        save_path = save_path)
        boxed_yhat = np.where(expanded_forms3D == 0, np.nan, np.array(results_maps['y_hat'])[:, :-vismargin, :-vismargin])
        title = 'All_mean_boxes'
        acumulated_yhat = np.nanmean(boxed_yhat, axis = 0)
        acumulated_yhat = (acumulated_yhat - np.nanmin(acumulated_yhat))/(np.nanmax(acumulated_yhat) - np.nanmin(acumulated_yhat))
        acumulated_masks = np.sum(results_maps['masks'], axis = 0)
        acumulated_labels = (np.sum(results_maps['labels'], axis = 0) != 0)
        plot_GDIS_map(title, 
                      acumulated_yhat, 
                      acumulated_masks[:-vismargin, :-vismargin], 
                      acumulated_labels[:-vismargin, :-vismargin], 
                      loader_test.dataset, dataset_config, save_path, 
                      bbox_contour = (np.sum(expanded_forms3D, axis = 0) != 0),
                      drought_id = None, set_colorbar = False, print_format = print_format)
        
        # >>> Plot confusion matrix
        #plot_GDIS_cm(results_maps['y_hat'], results_maps['labels'], results_maps['masks'], save_path, find_threshold = True, metric = "Youden's Index", print_format = print_format) # Youden's statistic === Youden's index
        #plot_GDIS_cm(results_maps['y_hat'], results_maps['labels'], results_maps['masks'], save_path, find_threshold = False, print_format = print_format)
        
        # >>> Some statistics regarding proportions of classes 
        n_drought_pixels  = np.sum(np.array(results_maps['labels']) == 1)
        n_nodrought_pixels = np.sum(np.array(results_maps['labels']) == 0)
        print('\nProportion of classes:')
        print('n_drought_pixels', n_drought_pixels)
        print('n_nodrought_pixels', n_nodrought_pixels)
        print('n_drought_pixels / n_nodrought_pixels', n_drought_pixels / n_nodrought_pixels)
        
        print('Process finished')
