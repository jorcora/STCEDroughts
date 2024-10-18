#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# GENERAL
import os

# PARSER
import argparse, yaml

# DATETIME
from datetime import datetime

def setup(filename):
    """Loads configuration yaml file as a dictionary

    :param filename: configuration file
    :type filename: yaml
    :return: configuration file
    :rtype: dict
    """    
    # Load YAML config file into a dict variable
    with open(filename) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config
    
def setup_experiment():
    """Gets the experimental, model and dataset configurations from the stored config. files.
    Modifies the stored configuration with the hyperparameters given through the command line. 
    Creates the folder structure for the experiment. 
    
    :return: condiguration files
    :rtype: dict
    """    
    # Parent path
    parent_path = '/home/jorcora/Location_Aware_AM'
    
    # Model and dataset configuration files
    dataset_config = setup(parent_path + '/configs/config_dataset.yaml')
    model_config = setup(parent_path + '/configs/config_model.yaml')
    
    # >>> Experimental setup (definition of hyperparams)
    general_parser = argparse.ArgumentParser(description="ArgParse")

    # General
    general_parser.add_argument('--name', default='noName', type=str)
    general_parser.add_argument('--accelerator', default='cpu', type=str)
    general_parser.add_argument('--print_format', default='pdf', type=str)

    # Type of model and fold
    general_parser.add_argument('--exp_id', default='3D', type=str)
    general_parser.add_argument('--fold_id', default='F1', type=str) 
    
    # -> SPX
    general_parser.add_argument('--SPX', default = model_config['SPX']['SPX'], type = int)
    general_parser.add_argument('--metric_eps_score', default = model_config['SPX']['metric_eps_score'], type = str) 
    general_parser.add_argument('--eps', default = model_config['SPX']['eps'], type = float)
    
    # Type of execution (fit/test/inference)
    general_parser.add_argument('--doFreeze', default=0, type=str)
    general_parser.add_argument('--doPretrained', default=0, type=int)
    general_parser.add_argument('--doFit', default=0, type=int)
    general_parser.add_argument('--doTest', default=0, type=int)
    general_parser.add_argument('--doInference', default=0, type=int)

    # >>> Experimental setup (update configuration files) 
    args = general_parser.parse_args()
    
    # Select config files specific to model and fold
    model_config['arch'].update(model_config['arch']['archs'][args.exp_id])
    del model_config['arch']['archs']
    
    dataset_config['GDIS'].update(dataset_config['GDIS']['folds'][args.fold_id])
    del dataset_config['GDIS']['folds']

    # Type of modification over the baseline model
    # -> SPX
    model_config['SPX']['SPX'] = args.SPX
    model_config['SPX']['metric_eps_score'] = args.metric_eps_score
    model_config['SPX']['eps'] = args.eps
    
    # >>> Create experiment config
    experiment_config = {}
    experiment_config['Arguments'] = vars(args)

    # Experiment project and name
    experiment_config['project'] = 'TEST'  
    experiment_config['date'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
    experiment_config['name'] = '_'.join([a + '-' + str(b) for a, b in zip(vars(args).keys(), vars(args).values()) if not 'do' in a]) 
    
    # Define and store paths
    experiment_config['parent_path'] = parent_path
    experiment_config['subparent_folder'] = os.path.join(experiment_config['parent_path'],'code/experiments')
    experiment_config['experiment_path'] = os.path.join(experiment_config['subparent_folder'], args.name)
    experiment_config['logs_path'] = os.path.join(experiment_config['experiment_path'], 'logs')
    experiment_config['results_path'] = os.path.join(experiment_config['experiment_path'], 'results')
    experiment_config['project_path'] = os.path.join(experiment_config['results_path'], experiment_config['project'])
    experiment_config['run_path'] = os.path.join(experiment_config['project_path'], experiment_config['name'])
    experiment_config['checkpoint_path'] = os.path.join(experiment_config['logs_path'], experiment_config['project'], experiment_config['name'])
    experiment_config['images_spx_path'] = os.path.join(experiment_config['run_path'], 'labels_superpixels')
    experiment_config['pretrained_model_path'] = ''
    
    # Check if a previous run with the same configuration exists 
    # if True get the path pointing to the best model,
    # this will allow us to perform testing/inference without training again 
    # this works if the hyperparameters are the same (except for the type of execution: 'do...')
    # if we also specify pretrained, it is going to change the name of the run
    # and activate the attention mechanism (if asked for)
    
    # Check if a previous config. file exist 
    if os.path.exists(experiment_config['run_path'] + '/experiment_config.yaml'):       
        with open(experiment_config['run_path'] + '/experiment_config.yaml', 'r') as file: 
                        
            # Get the path pointing to the checkpoint of the best model
            tmp = yaml.load(file, Loader=yaml.FullLoader) 
            experiment_config['pretrained_model_path'] = tmp['pretrained_model_path']
        
        if experiment_config['Arguments']['doPretrained']:
            print('Pretrained run')
            
            # modify the name of the run and load the pretrained experiment run path
            experiment_config['name'] += '_Pretrained'
            experiment_config['run_path'] = os.path.join(experiment_config['project_path'], experiment_config['name'])
            
            if experiment_config['Arguments']['doFit']:
                experiment_config['checkpoint_path'] = os.path.join(experiment_config['logs_path'], 
                                                                    experiment_config['project'], 
                                                                    experiment_config['name'])
            
    # Create folders
    if not os.path.isdir(experiment_config['experiment_path']):
           os.mkdir(experiment_config['experiment_path'])
           
    if not os.path.isdir(experiment_config['logs_path']):
           os.mkdir(experiment_config['logs_path'])
           
    if not os.path.isdir(experiment_config['results_path']):
           os.mkdir(experiment_config['results_path'])
    
    if not os.path.isdir(experiment_config['project_path']):
           os.mkdir(experiment_config['project_path'])
    
    if not os.path.isdir(experiment_config['run_path']):
           os.mkdir(experiment_config['run_path'])
           
    if not os.path.isdir(experiment_config['images_spx_path']):
           os.mkdir(experiment_config['images_spx_path'])
    
    # Save configs into the run folder
    with open(experiment_config['run_path'] + '/experiment_config.yaml', 'w') as file:
        yaml.dump(experiment_config, file)
    
    with open(experiment_config['run_path'] + '/dataset_config.yaml', 'w') as file:
        yaml.dump(dataset_config, file)
    
    with open(experiment_config['run_path'] + '/model_config.yaml', 'w') as file:
        yaml.dump(model_config, file)
    
    return experiment_config, dataset_config, model_config
    
def setup_experiment_evaluate():
    """Gets the experimental, model and dataset configurations from the stored config. files.
    Modifies the stored configuration with the hyperparameters given through the command line. 
    Creates the folder structure for the experiment. 

    :return: condiguration files
    :rtype: dict
    """ 
    # Parent path
    parent_path = '/home/jorcora/Location_Aware_AM'
    
    # Model and dataset configuration files
    dataset_config = setup(parent_path + '/configs/config_dataset.yaml')
    model_config = setup(parent_path + '/configs/config_model.yaml')
    
    # >>> Experimental setup (definition of hyperparams)
    general_parser = argparse.ArgumentParser(description="ArgParse")
    general_parser.add_argument('--print_format', default='pdf', type=str)
    general_parser.add_argument('--exp_id', default='3D', type=str)
    general_parser.add_argument('--fold_id', default='F1', type=str)
    general_parser.add_argument('--model_name', default='alpha', type=str)
    general_parser.add_argument('--ref_timescale', default=dataset_config['DInd']['ref_timescale'], type=str)  
    general_parser.add_argument('--commonMask', default=dataset_config['DInd']['commonMask'], type=bool)
    
    # >>> Experimental setup (update configuration files) 
    args = general_parser.parse_args()
    
    # Select config files specific to model
    model_config['arch'].update(model_config['arch']['archs'][args.exp_id])
    del model_config['arch']['archs']

    # Hyperparameters
    dataset_config['DInd']['ref_timescale'] = args.ref_timescale
    dataset_config['DInd']['commonMask'] = args.commonMask
    
    # >>> Create experiment config
    experiment_config = {}
    experiment_config['Arguments'] = vars(args)
    
    # Experiment project and name
    experiment_config['date'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
    experiment_config['name'] = 'Eval_' + '_'.join([a + '-' + str(b) for a, b in zip(vars(args).keys(), vars(args).values())])
    
    # Define and store paths
    experiment_config['parent_path'] = parent_path
    experiment_config['subparent_folder'] = os.path.join(experiment_config['parent_path'],'code/experiments_evaluate')
    experiment_config['run_path'] = os.path.join(experiment_config['subparent_folder'], experiment_config['name'])
    experiment_config[f'pretrained_model_path_F1_{args.model_name}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F1_{args.model_name}.pkl')
    experiment_config[f'pretrained_model_path_F3_{args.model_name}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F3_{args.model_name}.pkl')
    experiment_config[f'pretrained_model_path_F5_{args.model_name}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F5_{args.model_name}.pkl')
    
    # Create folders
    if not os.path.isdir(experiment_config['run_path']):
           os.mkdir(experiment_config['run_path'])
    
    # Save configs into the run folder
    with open(experiment_config['run_path'] + '/experiment_config.yaml', 'w') as file:
        yaml.dump(experiment_config, file)
    
    with open(experiment_config['run_path'] + '/dataset_config.yaml', 'w') as file:
        yaml.dump(dataset_config, file)
    
    with open(experiment_config['run_path'] + '/model_config.yaml', 'w') as file:
        yaml.dump(model_config, file)
    
    return experiment_config, dataset_config, model_config

def setup_experiment_compare():
    """Gets the experimental, model and dataset configurations from the stored config. files.
    Modifies the stored configuration with the hyperparameters given through the command line. 
    Creates the folder structure for the experiment. 

    :return: condiguration files
    :rtype: dict
    """     
    # Parent path
    parent_path = '/home/jorcora/Location_Aware_AM'
    
    # Model and dataset configuration files
    dataset_config = setup(parent_path + '/configs/config_dataset.yaml')
    model_config = setup(parent_path + '/configs/config_model.yaml')
    
    # >>> Experimental setup (definition of hyperparams)
    general_parser = argparse.ArgumentParser(description="ArgParse")
    general_parser.add_argument('--exp_id', default='3D', type=str)
    general_parser.add_argument('--fold_id', default='FA', type=str)
    general_parser.add_argument('--model_name_1', default='alpha', type=str)
    general_parser.add_argument('--model_name_2', default='omega', type=str)
    general_parser.add_argument('--normalize_maps', default=1, type=int)
    general_parser.add_argument('--ref_timescale', default=dataset_config['DInd']['ref_timescale'], type=str)  
    general_parser.add_argument('--print_format', default='pdf', type=str)
    
    # >>> Experimental setup (update configuration files) 
    args = general_parser.parse_args()
    
    # Select config files specific to model
    model_config['arch'].update(model_config['arch']['archs'][args.exp_id])
    del model_config['arch']['archs']
    
    # Hyperparameters 
    dataset_config['DInd']['ref_timescale'] = args.ref_timescale
    
    # >>> Create experiment config
    experiment_config = {}
    experiment_config['Arguments'] = vars(args)
    
    # Experiment project and name
    experiment_config['date'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
    experiment_config['name'] = 'Comp_' + '_'.join([a + '-' + str(b) for a, b in zip(vars(args).keys(), vars(args).values())])
    
    # Define and store paths
    experiment_config['parent_path'] = parent_path
    experiment_config['subparent_folder'] = os.path.join(experiment_config['parent_path'],'code/experiments_evaluate')
    experiment_config['run_path'] = os.path.join(experiment_config['subparent_folder'], experiment_config['name'])
    experiment_config[f'pretrained_model_path_F1_{args.model_name_1}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F1_{args.model_name_1}.pkl')
    experiment_config[f'pretrained_model_path_F3_{args.model_name_1}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F3_{args.model_name_1}.pkl')
    experiment_config[f'pretrained_model_path_F5_{args.model_name_1}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F5_{args.model_name_1}.pkl')
    
    experiment_config[f'pretrained_model_path_F1_{args.model_name_2}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F1_{args.model_name_2}.pkl')
    experiment_config[f'pretrained_model_path_F3_{args.model_name_2}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F3_{args.model_name_2}.pkl')
    experiment_config[f'pretrained_model_path_F5_{args.model_name_2}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F5_{args.model_name_2}.pkl')
    
    # Create folders
    if not os.path.isdir(experiment_config['run_path']):
           os.mkdir(experiment_config['run_path'])
    
    # Save configs into the run folder
    with open(experiment_config['run_path'] + '/experiment_config.yaml', 'w') as file:
        yaml.dump(experiment_config, file)
    
    with open(experiment_config['run_path'] + '/dataset_config.yaml', 'w') as file:
        yaml.dump(dataset_config, file)
    
    with open(experiment_config['run_path'] + '/model_config.yaml', 'w') as file:
        yaml.dump(model_config, file)
    
    return experiment_config, dataset_config, model_config

def setup_experiment_evaluate_context():
    """Gets the experimental, model and dataset configurations from the stored config. files.
    Modifies the stored configuration with the hyperparameters given through the command line. 
    Creates the folder structure for the experiment. 

    :return: condiguration files
    :rtype: dict
    """     
    # Parent path
    parent_path = '/home/jorcora/Location_Aware_AM'

    # Model and dataset configuration files
    dataset_config = setup(parent_path + '/configs/config_dataset.yaml')
    model_config = setup(parent_path + '/configs/config_model.yaml')
    
    # >>> Experimental setup (definition of hyperparams)
    general_parser = argparse.ArgumentParser(description="ArgParse")
    general_parser.add_argument('--exp_id', default='3D', type=str)
    general_parser.add_argument('--fold_id', default='F3', type=str)
    general_parser.add_argument('--model_name_1', default='alpha_Baseline', type=str)
    general_parser.add_argument('--model_name_2', default='omega_SPX', type=str)
    general_parser.add_argument('--ref_timescale', default=dataset_config['DInd']['ref_timescale'], type=str)  
    general_parser.add_argument('--commonMask', default=dataset_config['DInd']['commonMask'], type=bool)
    general_parser.add_argument('--print_format', default='pdf', type=str)
    
    # >>> Experimental setup (update configuration files) 
    args = general_parser.parse_args()
    
    # Select config files specific to model
    model_config['arch'].update(model_config['arch']['archs'][args.exp_id])
    del model_config['arch']['archs']

    # Hyperparameters
    dataset_config['DInd']['ref_timescale'] = args.ref_timescale
    dataset_config['DInd']['commonMask'] = args.commonMask
    
    # >>> Create experiment config
    experiment_config = {}
    experiment_config['Arguments'] = vars(args)
    
    # Experiment project and name
    experiment_config['date'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
    experiment_config['name'] = 'EvalContext_' + '_'.join([a + '-' + str(b) for a, b in zip(vars(args).keys(), vars(args).values())])
    
    # Define and store paths
    experiment_config['parent_path'] = parent_path
    experiment_config['subparent_folder'] = os.path.join(experiment_config['parent_path'],'code/experiments_evaluate')
    experiment_config['run_path'] = os.path.join(experiment_config['subparent_folder'], experiment_config['name'])
    experiment_config[f'pretrained_model_path_F1_{args.model_name_1}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F1_{args.model_name_1}.pkl')
    experiment_config[f'pretrained_model_path_F3_{args.model_name_1}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F3_{args.model_name_1}.pkl')
    experiment_config[f'pretrained_model_path_F5_{args.model_name_1}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F5_{args.model_name_1}.pkl')
    
    experiment_config[f'pretrained_model_path_F1_{args.model_name_2}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F1_{args.model_name_2}.pkl')
    experiment_config[f'pretrained_model_path_F3_{args.model_name_2}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F3_{args.model_name_2}.pkl')
    experiment_config[f'pretrained_model_path_F5_{args.model_name_2}'] = os.path.join(experiment_config['subparent_folder'], f'evaluate_results_F5_{args.model_name_2}.pkl')
    
    # Create folders
    if not os.path.isdir(experiment_config['run_path']):
           os.mkdir(experiment_config['run_path'])
    
    # Save configs into the run folder
    with open(experiment_config['run_path'] + '/experiment_config.yaml', 'w') as file:
        yaml.dump(experiment_config, file)
    
    with open(experiment_config['run_path'] + '/dataset_config.yaml', 'w') as file:
        yaml.dump(dataset_config, file)
    
    with open(experiment_config['run_path'] + '/model_config.yaml', 'w') as file:
        yaml.dump(model_config, file)
    
    return experiment_config, dataset_config, model_config
    

