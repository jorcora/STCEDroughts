#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# GENERAL
import sys

# DATASET
from dataset.GDIS_evaluate import EvalGDIS

# WARNING
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

# UTILS
from utils import setup_experiment_evaluate
from dataset.dataset_utils import PrepRES

if __name__ == '__main__':

    # Configurations
    experiment_config, dataset_config, model_config = setup_experiment_evaluate()
    
    # Data preparation
    prepRES = PrepRES(dataset_config, experiment_config)
    
    # Load model results and indices
    model, indices = prepRES._get_results_indices_(model_type = experiment_config['Arguments']['model_name'])
    
    # Class for evaluating 
    evalGDIS = EvalGDIS(experiment_config)
        
    # External file to save results
    sys.stdout = open(experiment_config['run_path'] + '/Results.txt', 'w')
    print('\nExperiment:', experiment_config['name'])
    
    # >>> Evaluate for each drought index
    for drought_index_name in list(prepRES.indices.keys()):
        evalGDIS._evaluation(['model', drought_index_name], [model, indices])
        
    # >>> Evaluate only the model 
    # (evaluation independent from indices locations -> evaluation only on model locations)
    if dataset_config['DInd']['ref_timescale'] == 'model':
        
        print('Evaluating the model only on model locations')
        evalGDIS._evaluation(['model'], [model])

    # >>> Save predictions on each location and corresponding labels
    evalGDIS._save_prediccions()
    
    # >>> Plot all AUROC curves (each prediction is over its own location)
    evalGDIS._plot_AUROCcurves()
    
    print('\nProcess finished')
