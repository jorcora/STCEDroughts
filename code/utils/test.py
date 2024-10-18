#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY 
import numpy as np

# SKLEARN 
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from utils import crop_variable, cm_measures

def cube_metrics(output, labels, masks, map_dominant_context, context_names):
    """Computes metrics at cube level

    :param output: predictions of the model
    :type output: np.ndarray
    :param labels: target variable
    :type labels: np.ndarray
    :param masks: where we have input data or not (1==data)
    :type masks: np.ndarray
    :param map_dominant_context: map of the dominant land use
    :type map_dominant_context: np.ndarray
    :param context_names: names of the land use types
    :type context_names: str
    :return: metrics computed 
    :rtype: np.ndarrays
    """    
    # flatten the variables
    outputs = output.flatten()
    labels = labels.flatten()
    masks = masks.flatten()
    map_dominant_context = map_dominant_context.flatten()
    
    # Apply the mask
    outputs = outputs[masks]
    labels = labels[masks]
    map_dominant_context = map_dominant_context[masks]
    
    # Predefine results
    auroc = np.nan
    aucpr = np.nan
    f1 = np.nan
    perfmeasures = {'TP': np.nan, 'TN': np.nan, 'FP': np.nan, 'FN': np.nan} # has to be predefined in case we do not have true labels in the cube. Will be replaced if we have true labels
    auroc_context = {key:np.nan for key in context_names} 
    
    # Get/compute the variables 
    if labels.any(): 
        
        auroc = roc_auc_score(labels, outputs)
        aucpr = average_precision_score(labels, outputs)
        f1 = f1_score(labels, outputs.round(), zero_division = 'warn')
        perfmeasures = cm_measures(labels, outputs.round())
        
        for id_context, name_context in enumerate(context_names):
            context_locations = (map_dominant_context == id_context)
            
            if labels[context_locations].any():
                auroc_context[name_context] = roc_auc_score(labels[context_locations], 
                                                            outputs[context_locations])
                
    return auroc, aucpr, f1, perfmeasures, auroc_context

def instance_metrics(output, labels, masks, map_dominant_context,
                     context_names, mask_event_locations, mask_event_ids, idt):
    """Computes metrics at instance level

    :param output: predictions of the model
    :type output: np.ndarray
    :param labels: target variable
    :type labels: np.ndarray
    :param masks: where we have input data or not (1==data)
    :type masks: np.ndarray
    :param map_dominant_context: map of the dominant land use
    :type map_dominant_context: np.ndarray
    :param context_names: names of the land use types
    :type context_names: str
    :param mask_event_locations: locations of the drought events
    :type mask_event_locations: np.ndarray
    :param mask_event_ids: ids of the drought events
    :type mask_event_ids: np.ndarrays
    :param idt: identification to define the temporal instance
    :type idt: int
    :return: metrics computed 
    :rtype: np.ndarray
    """    
    # instance variables
    outputs = output[0][0][idt].flatten()
    labels = labels[0][0][idt].flatten()
    masks = masks[0][0][idt].flatten()
    map_dominant_context = map_dominant_context[0][0][idt].flatten()
    mask_event_locations = mask_event_locations[0][:, idt]
    
    # Apply the mask
    outputs = outputs[masks]
    labels = labels[masks]
    map_dominant_context = map_dominant_context[masks]
    
    # Predefine results
    auroc = np.nan
    auroc_context = {key:np.nan for key in context_names} 
    
    signal_outputs = {key:np.nan for key in mask_event_ids} 
    signal_labels = {key:np.nan for key in mask_event_ids} 
    signal_outputs_context = {key:{key:np.nan for key in mask_event_ids} for key in context_names}
    signal_labels_context = {key:{key:np.nan for key in mask_event_ids} for key in context_names}
    
    # Get/compute the variables
    # AUROC
    if labels.any():
        
        # general
        auroc = roc_auc_score(labels, outputs)
        
        # wrt context
        for id_context, name_context in enumerate(context_names):
            context_locations = (map_dominant_context == id_context)
            
            if labels[context_locations].any():
                auroc_context[name_context] = roc_auc_score(labels[context_locations], 
                                                            outputs[context_locations])
    
    # Probs (signals)
    for event_id in mask_event_ids:
        location_event = (mask_event_locations == event_id).any(axis = 0).flatten()
        location_event = location_event[masks]
        
        if location_event.any():
            
            signal_outputs[event_id] = np.mean(outputs[location_event])
            signal_labels[event_id] = labels[location_event].any() * 1.0
            
            # Probs (signals) wrt context
            for id_context, name_context in enumerate(context_names): 
                event_context_locations = np.logical_and(location_event, map_dominant_context == id_context)
                
                if not len(labels[event_context_locations]) == 0:
                    signal_outputs_context[name_context][event_id] = np.mean(outputs[event_context_locations])
                    signal_labels_context[name_context][event_id] = labels[event_context_locations].any()    
        else:
            signal_outputs[event_id] = 0
            signal_labels[event_id] = 0
            
    return auroc, auroc_context, signal_outputs, signal_labels, signal_outputs_context, signal_labels_context

def measures_test(res, dataset_config, masks, map_dominant_context, mask_event_ids, mask_event_locations):
    """Computes extended measures for testing at cube and instance level

    :param output: predictions of the model
    :type output: torch.Tensor
    :param labels: target variable
    :type labels: torch.Tensor
    :param masks: where we have input data or not (1==data)
    :type masks: torch.Tensor
    :param map_dominant_context: map of the dominant land use
    :type map_dominant_context: torch.Tensor
    :param context_names: names of the land use types
    :type context_names: str
    :param mask_event_locations: locations of the drought events
    :type mask_event_locations: torch.Tensor
    :return: metrics computed 
    :rtype: np.ndarray
    """    
    # Correct for the empty border that is added in the superior and left side of the inputs
    # during the reconstruction step of the map
    size_in = list(eval(dataset_config['GDIS']['input_size']))
    size_out = list(eval(dataset_config['GDIS']['output_size']))
    lost_border = int((size_in[0] - size_out[0])/2)
    idt = dataset_config['GDIS']['idt']
    
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
    mask_event_ids = mask_event_ids.cpu().numpy()[0]
    
    # Move variables to the cpu and turn them into numpy arrays
    m_outputs = res['output'].cpu().numpy()
    m_labels = res['labels'].cpu().numpy()
    m_masks = masks.cpu().numpy()
    m_map_dominant_context = map_dominant_context.cpu().numpy()
    m_mask_event_locations = mask_event_locations.cpu().numpy()
    
    # Context names
    context_names = np.array(dataset_config['HWSD']['features'])[dataset_config['HWSD']['features_selected']]
    
    # --------- CUBE ------------
    cube_res = cube_metrics(m_outputs, m_labels, m_masks, m_map_dominant_context, context_names)
    
    # ---------- INSTANCES ----------
    instance_res = instance_metrics(m_outputs, m_labels, m_masks, m_map_dominant_context, context_names, m_mask_event_locations, mask_event_ids, idt)
    
    outputs_masked = res['output'][0][0][idt].flatten()
    labels_masked = res['labels'][0][0][idt].flatten()
    temp_masks = masks[0][0][idt].flatten()
    
    outputs_masked = outputs_masked[temp_masks]
    labels_masked = labels_masked[temp_masks]
    
    return {'loss': res['loss'], 
            'auroc_step': cube_res[0], 'aucpr_step': cube_res[1], 'f1_step': cube_res[2],
            'TP': cube_res[3]['TP'], 'TN': cube_res[3]['TN'],
            'FP': cube_res[3]['FP'], 'FN': cube_res[3]['FN'],
            'test_auroc_step_context': cube_res[4],
            'instance_test_auroc': instance_res[0],
            'instance_test_auroc_step_context': instance_res[1],
            'test_signal_outputs': instance_res[2],
            'test_signal_labels':instance_res[3],
            'test_signal_step_context_outputs': instance_res[4],
            'test_signal_step_context_labels': instance_res[5],
            'mask_event_ids': mask_event_ids, 
            'outputs_masked': outputs_masked,
            'labels_masked': labels_masked}
