#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# GENERAL
import sys

# DATASET
from dataset.GDIS_evaluate_context import EvalGDIScontext

# WARNING
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

# UTILS
from utils import setup_experiment_evaluate_context
from dataset.dataset_utils import PrepRES

if __name__ == '__main__':

    # Configurations
    experiment_config, dataset_config, model_config = setup_experiment_evaluate_context()
    
    # Data preparation
    prepRES = PrepRES(dataset_config, experiment_config)
    
    # Load model results and indices
    alpha, indices = prepRES._get_results_indices_(model_type = experiment_config['Arguments']['model_name_1'])
    omega, _ = prepRES._get_results_indices_(model_type = experiment_config['Arguments']['model_name_2'])
    
    # Class for evaluating 
    evalGDIScontext = EvalGDIScontext(experiment_config)
        
    # External file to save results
    sys.stdout = open(experiment_config['run_path'] + '/Results.txt', 'w')
    print('\nExperiment:', experiment_config['name'])
    
    # >>> Evaluate for each drought index
    for drought_index_name in list(prepRES.indices.keys()):
        
        evalGDIScontext._evaluation_wrt_context(['alpha', drought_index_name], [alpha, indices])
        evalGDIScontext._evaluation_wrt_context(['omega', drought_index_name], [omega, indices])
        
    model_names = list(prepRES.indices.keys()) + ['alpha', 'omega']
    evalGDIScontext._plot_groupbarplot(model_names, evalGDIScontext.merged_res)
    
    # >>> Save predictions on each location and corresponding labels
    evalGDIScontext._save_merged_res(model_names)

    # >>> Evaluate only the model 
    # (evaluation independent from indices locations -> evaluation only on model locations)
    if dataset_config['DInd']['ref_timescale'] == 'model':
        
        print('Evaluating the model only on model locations')
        evalGDIScontext._evaluation_wrt_context(['alpha'], [alpha])
        evalGDIScontext._evaluation_wrt_context(['omega'], [omega])
        
    model_names = ['alpha', 'omega']
    evalGDIScontext._plot_groupbarplot(model_names, evalGDIScontext.merged_res)

    # >>> Save predictions on each location and corresponding labels
    evalGDIScontext._save_merged_res(model_names)

    print('\nProcess finished')
    

