#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# GENERAL
import os, re

# YAML
import yaml

# PYTORCH
import torch
import torch.nn as nn

# NUMPY
import numpy as np

def check_init(MY_MODEL, model, experiment_config, dataset_config, model_config):
    """Checks that the initialization between the current model and 
    the stored reference is the same. Creates the ref. if not found. 

    :param MY_MODEL: model class (non initialized object)
    :type MY_MODEL: pytorch_lightning.LightningModule
    :param model: model class (initialized object)
    :type model: pytorch_lightning.LightningModule
    :param experiment_config: configuration of the current experiment
    :type experiment_config: dict
    :param dataset_config: configuration of the dataset
    :type dataset_config: dict
    :param model_config: configuration of the model
    :type model_config: dict
    """
    filename = experiment_config['parent_path'] + '/code/model/arch/' + model_config['arch']['model_init_file'] 
    if os.path.isfile(filename):
    
        # Load ref.  
        ref = MY_MODEL(experiment_config, dataset_config, model_config)
        ref.load_state_dict(torch.load(filename))

        # Check each layer
        match = True
        for key_item_1, key_item_2 in zip(model.state_dict().items(), ref.state_dict().items()):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                
                match = False
                print('Non match at:', key_item_1[0], 
                      '\nMean diff. value:', torch.mean(key_item_1[1] - key_item_2[1]))
                
        print('-Initial state of the model matches the ref.:', match)
        print('-Name of the init file:', model_config['arch']['model_init_file'])
        assert match, 'Created model and ref. do not have the same init'
    
    else:
        
        print('-Saving initial state of the model (=== ref.)')
        torch.save(model.state_dict(), filename)

def define_model(MY_MODEL, experiment_config, dataset_config, model_config):
    """Defines the model 

    :param MY_MODEL: model class (non initialized object)
    :type MY_MODEL: pytorch_lightning.LightningModule
    :param experiment_config: configuration of the current experiment
    :type experiment_config: dict
    :param dataset_config: configuration of the dataset
    :type dataset_config: dict
    :param model_config: configuration of the model
    :type model_config: dict
    :return: model class (initialized object)
    :rtype: pytorch_lightning.LightningModule
    """    
    print('\nGetting the model...')
    
    # Create model and check: init == init_reference
    model = MY_MODEL(experiment_config, dataset_config, model_config)
    check_init(MY_MODEL, model, experiment_config, dataset_config, model_config)
    
    # Load a pretrained model if true
    if experiment_config['Arguments']['doPretrained']:
        model = load_pretrained(MY_MODEL, model, experiment_config)
                
    return model 

def load_pretrained(MY_MODEL, model, experiment_config):
    """Loads model from checkpoint. 
    If activating AM, the encoder layers are freezed

    :param MY_MODEL: model class (non initialized object)
    :type MY_MODEL: pytorch_lightning.LightningModule
    :param model: model class (initialized object)
    :type model: pytorch_lightning.LightningModule
    :param experiment_config: configuration of the current experiment
    :type experiment_config: dict
    :return: Model with the encoder and decoder part pretrained and freezed or 
    all pretrained and non freeezed. 
    :rtype: pytorch_lightning.LightningModule
    """    
    # Load pretrained model
    pretrained_model = MY_MODEL.load_from_checkpoint(experiment_config['pretrained_model_path'])
    if experiment_config['Arguments']['doFreeze']:

        # Update selected layers with prelearned parameter values and freeze
        print('Freezing Encoder-Decoder layers')
        for layer_name, layer in model.model.named_modules():
            if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.BatchNorm3d):
                if 'encoder' in layer_name or 'decoder' in layer_name:
                    
                    # Copy parameters values
                    getattr(model.model, layer_name).weight = getattr(pretrained_model.model, layer_name).weight
                    getattr(model.model, layer_name).bias = getattr(pretrained_model.model, layer_name).bias
                    
                    # Freeze layers
                    getattr(model.model, layer_name).weight.requires_grad = False
                    getattr(model.model, layer_name).bias.requires_grad = False 
        
        return model 
                
    else: 
        return pretrained_model

def save_model_reference(experiment_config, model_config): 
    """Searches for the best model according to the defined metric 
    and saves the corresponding path and name into the experiment config

    :param experiment_config: configuration of the current experiment
    :type experiment_config: dict
    :param model_config: configuration of the model
    :type model_config: dict
    :return: configuration of the current experiment updated
    :rtype: dict
    """
    print("\nSaving best model reference")
    print(f"Selected wrt {model_config['trainer']['mselection']} at its max value")
    
    # Acumulate the metrics of all checkpoints
    
    list_checkpoints = [filename for filename in os.listdir(experiment_config['checkpoint_path']) 
                        if model_config['trainer']['mselection'] in filename]

    metric_values = [float(
        re.split('[-=]', filename)[re.split('[-=]', filename).index(model_config['trainer']['mselection']) + 1])
        for filename in list_checkpoints]
    metric_values = np.where(np.isfinite(metric_values), metric_values, np.nan)
    
    # Get the best metric and corresponding model    
    idx_best_value = np.nanargmax(metric_values)
    best_model = list_checkpoints[idx_best_value]
    print('-Best model:', best_model)
    
    # Define and update the experiment config with the best model path
    experiment_config['pretrained_model_path'] = os.path.join(experiment_config['checkpoint_path'], best_model)
    with open(experiment_config['run_path'] + '/experiment_config.yaml', 'w') as f:
        yaml.dump(experiment_config, f)
    
    return experiment_config
