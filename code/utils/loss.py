#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# PYTORCH
import torch
import torch.nn.functional as F

def scores_focal_loss(model_config, p, targets):
    """Compute the focal loss

    :param model_config: model configuration
    :type model_config: dict
    :param p: probabilities (outputs of the model after a sigmoid)
    :type p: torch.Tensor
    :param targets: target variable 
    :type targets: torch.Tensor
    :return: focal loss scores
    :rtype: torch.Tensor
    """    
    ce_loss = F.binary_cross_entropy(p, targets, reduction = "none")
    p_t = p * targets + (1 - p) * (1 - targets)
    gamma = model_config['optimizer']['loss']['gamma']
    loss = ce_loss * ((1 - p_t) ** gamma)
        
    return loss

def combine_objects(obj_ori, obj_spx, eps = 0.5):
    """Convex combination of two objects given a mixing factor
    """    
    return ((1 - eps) * obj_ori + eps * obj_spx).float()

def compute_loss(model_config, output, labels, masks, 
                 spx_labels = None, eps = 0.5):
    """Computes the loss

    :param model_config: model configuration
    :type model_config: dict
    :param output: probabilities (outputs of the model after a sigmoid)
    :type output: torch.Tensor
    :param labels: target variable 
    :type labels: torch.Tensor
    :param masks: locations where we have input data, (1 == data, 0 == no data)
    :type masks: torch.Tensor
    :param spx_labels: spx labels, defaults to None
    :type spx_labels: torch.Tensor, optional
    :param eps: mixing factor, defaults to 0.5
    :type eps: float, optional
    :return: loss scores
    :rtype: torch.Tensor
    """    
    # Combine either losses or targets   
    if spx_labels == None:
        loss = scores_focal_loss(model_config, output, labels)
    
    else:
        if model_config['SPX']['combined_labels']:

            new_labels = combine_objects(labels, spx_labels, eps = eps)
            loss = scores_focal_loss(model_config, output, new_labels)
            
        if model_config['SPX']['combined_losses']:

            loss_ori = scores_focal_loss(model_config, output, labels)
            loss_spx = scores_focal_loss(model_config, output, spx_labels)
            loss = combine_objects(loss_ori, loss_spx, eps = eps)
    
    # Perform balancing and other corrections
    # Apply correction if there is at least a pixel of drought in the original labels
    if labels.any(): 
        
        # Compute w_imbalance factor: 
        # s_x = 1 / (p_x * N_c), where N_c: = number of classes, p_x: = prob of class_x 
        # Considering p_x defined as a relative frequency (empirical prob. given the sample)=> p_x = n_x / N_total
        # That is: s_x = N_total / (2 * n_x)
        corr_effective_false = 0
        if torch.sum(labels == 0) != 0: # prevent this factor if it goes to infinity (torch.sum(labels == 0) = 0)
            corr_effective_false = labels.numel()/(2 * torch.sum(labels == 0)) 
        corr_effective_true = labels.numel()/(2 * torch.sum(labels == 1)) 

        # Apply the imbalance
        w_imbalance = torch.ones_like(labels) 
        w_imbalance[labels == 0] = corr_effective_false
        w_imbalance[labels == 1] = corr_effective_true
        
        # Apply the imbalance factor
        loss *= w_imbalance 
    
    # Correct for locations where we don't have values (masks: 1 := data, 0 := no_data) 
    loss[masks == 0] = 0
        
    # Mean without the left out values 
    loss = loss.sum() / (torch.sum(masks) + 1e-12)
    
    return loss

