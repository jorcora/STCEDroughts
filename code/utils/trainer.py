#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# GENERAL
import os

# PYTORCH LIGHTNING
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

# WANDB
import wandb
os.environ['WANDB_START_METHOD'] ="thread"
os.environ['WANDB__SERVICE_WAIT'] = '300'
wandb.login()

def define_callbacks(experiment_config, model_config):
    """Defines the callbacks of the trainer

    :param experiment_config: configuration of the current experiment
    :type experiment_config: dict
    :param model_config: configuration of the model
    :type model_config: dict
    :return: callbacks
    :rtype: pytorch_lightning.callbacks
    """    
    # checkpoint
    checkpoint_callback = ModelCheckpoint(dirpath = os.path.join(experiment_config['logs_path'], experiment_config['project'], experiment_config['name']),
                                          filename = ('{epoch}' + '-{step}' + 
                                                      '-{avg_train_' + model_config['trainer']['monitor'] + ':.3f}' +
                                                      '-{avg_val_' + model_config['trainer']['monitor'] + ':.3f}' + 
                                                      '-{avg_val_loss:.3f}' +
                                                      '-{mselection:.3f}' + 
                                                      '-0'), # the epoch is from training and the step is global (accumulated training steps)
                                          monitor = 'avg_val_' + model_config['trainer']['monitor'], 
                                          mode = 'max', 
                                          save_weights_only = True, #False (default),
                                          save_on_train_epoch_end = False, # change this to False to save validation checkpoints (default is None => True)
                                          save_last = True, save_top_k = -1) #save_top_k == -1, all models are saved (take care of memory issues) (5 default)
    # early_stopping                                   
    early_stopping = EarlyStopping(monitor = 'avg_val_' + model_config['trainer']['monitor'], #'avg_val_loss', #
                                   patience = model_config['trainer']['early_stop'], # number of validation checks with no improvement  
                                   check_on_train_epoch_end = False, 
                                   mode = 'max', 
                                   min_delta = 0.0,
                                   strict = True)
    # progress_bar
    pbar = TQDMProgressBar(refresh_rate = 10)
    
    return [checkpoint_callback, early_stopping, pbar] 
    
def define_logger(experiment_config):
    """Definition of the wandb logger for tracking the training and evaluation

    :param experiment_config: configuration of the current experiment
    :type experiment_config: dict
    :return: wandb logger
    :rtype: pytorch_lightning.loggers
    """    
    # online logger
    wandb_logger = pl_loggers.WandbLogger(name = experiment_config['name'],
                                          save_dir = experiment_config['logs_path'],
                                          project = experiment_config['project'], 
                                          id = experiment_config['date'],
                                          **{'tags': experiment_config['name'].split('_'), 'save_code': True, 'reinit': True, 'mode': 'online'})
                                          
    return wandb_logger
    
def define_trainer(experiment_config, model_config):
    """Definition of the Trainer class of pytorch lightning

    :param experiment_config: configuration of the current experiment
    :type experiment_config: dict
    :param model_config: configuration of the model
    :type model_config: dict
    :return: trainer class
    :rtype: pytorch_lightning.Trainer
    """    
    # callbacks
    callbacks = define_callbacks(experiment_config, model_config)
    
    # logger
    logger = define_logger(experiment_config)
    
    # trainer
    trainer = pl.Trainer(num_sanity_val_steps = 0, 
                         accelerator = experiment_config['Arguments']['accelerator'],
                         devices = 'auto',
                         callbacks = callbacks, 
                         logger = logger, 
                         max_epochs = model_config['trainer']['epochs'], 
                         reload_dataloaders_every_n_epochs = 0, 
                         deterministic = 'warn', 
                         val_check_interval = 0.25)
                         #track_grad_norm = 1)  
                         # sync_batchnorm = False, # set this to True if using multi GPU
                         
    return trainer
