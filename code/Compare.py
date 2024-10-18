#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# GENERAL
import sys

# DATASET
from dataset.GDIS_compare import CompGDIS

# WARNING
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

# UTILS
from utils import setup_experiment_compare
from dataset.dataset_utils import PrepRES

if __name__ == '__main__':
    
    # Configurations
    experiment_config, dataset_config, model_config = setup_experiment_compare()
    
    # Data preparation
    prepRES = PrepRES(dataset_config, experiment_config)
    
    # Load models results 
    model_alpha, _ = prepRES._get_results_indices_(model_type = experiment_config['Arguments']['model_name_1'])
    model_omega, _ = prepRES._get_results_indices_(model_type = experiment_config['Arguments']['model_name_2'])
    
    # External file to save results
    sys.stdout = open(experiment_config['run_path'] + '/Results.txt', 'w')
    print('\nExperiment:', experiment_config['name'])
    
    # Class for comparison 
    compGDIS = CompGDIS(dataset_config, experiment_config)
    
    # >>> Compare models
    compGDIS._compare_models_(model_alpha, model_omega, prepRES.time_ref, 
                              normalize_maps = experiment_config['Arguments']['normalize_maps'], 
                              print_format = experiment_config['Arguments']['print_format'])
    
    print('Process finished')
