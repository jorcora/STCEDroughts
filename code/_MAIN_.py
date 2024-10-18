#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi cortés-andrés
"""

# GENERAL
import sys

# TIME
import time

# PYTORCH LIGHTNING
import pytorch_lightning as pl
pl.seed_everything(0, workers = True) # sets the seed for numpy, general (random) and torch processes in gpu and cpu 

# PYTORCH
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') 
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

# DATASET and MODEL 
from dataset.GDIS_dataset_indices import GDIS
from model.GDIS_model_indices import MY_MODEL

# UTILS
from utils import (setup_experiment, 
                   define_dataloaders, 
                   define_model, 
                   define_trainer, 
                   save_model_reference, 
                   do_inference)

# WARNINGS
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

if __name__ == '__main__':  
    
    # Configurations
    experiment_config, dataset_config, model_config = setup_experiment()
    
    # Dataloaders
    loader_train, loader_val, loader_test, dataset_config = define_dataloaders(GDIS, experiment_config, dataset_config)     
    
    # Model
    model = define_model(MY_MODEL, experiment_config, dataset_config, model_config) 
        
    # Trainer
    trainer = define_trainer(experiment_config, model_config) 
    
    # Fit
    if experiment_config['Arguments']['doFit']:
        
        start_time = time.time()
        trainer.fit(model, train_dataloaders = loader_train, val_dataloaders = loader_val)
        experiment_config['train_time_min'] = round((time.time() - start_time) / 60, 2)
        print('Train time:', experiment_config['train_time_min'], 'min')

    # Select best model 
    experiment_config = save_model_reference(experiment_config, model_config)
    
    # Test and inference
    if (experiment_config['Arguments']['doTest'] or 
        experiment_config['Arguments']['doInference']):
         
        # External file to save results
        sys.stdout = open(experiment_config['run_path'] + '/Results.txt', 'w')
        print('Best model:', experiment_config['pretrained_model_path'], '\n')
        
        # Load model
        model = MY_MODEL.load_from_checkpoint(experiment_config['pretrained_model_path'])
        
        # Testing
        if experiment_config['Arguments']['doTest']:
            trainer.test(model, dataloaders = loader_test)
            
        # Inference
        if experiment_config['Arguments']['doInference']:
            do_inference(model, loader_test, dataset_config, experiment_config['run_path'],
                         print_format = experiment_config['Arguments']['print_format'])

    print('\nProcess finished')
