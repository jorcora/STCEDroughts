#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY  
import numpy as np

# PYTORCH
from torch.utils.data import DataLoader
from torch.utils.data import sampler

# SCIPY
from scipy.stats import ttest_ind

# ITERTOOLS
from itertools import product

# SEABORN
import seaborn as sns

# MATPLOTLIB
import matplotlib.pyplot as plt
        
def visualize_dist(dist_train, dist_val, id_samples_train, id_samples_val, savepath):
    """Plots the distribution of drought percentages 
    for all the possible samples and the samples selected 

    :param dist_train: train distribution
    :type dist_train: numpy.ndarray
    :param dist_val: val distribution
    :type dist_val: numpy.ndarray
    :param id_samples_train: samples to take for training
    :type id_samples_train: list
    :param id_samples_val: samples to take for valitation
    :type id_samples_val: list
    :param savepath: path to save
    :type savepath: str
    """
    fig, ax = plt.subplots(1,2)
    
    # Possible samples
    sns.kdeplot(data = dist_train, color = 'red', label = 'train', fill = True, ax = ax[0]) 
    sns.kdeplot(data = dist_val, color = 'blue', label = 'val', fill = True, ax = ax[0])
    ax[0].legend()
    ax[0].set_title('Original')
    ax[0].set_xlabel('% drought pixels (in samples)')
    
    # Samples taken
    sns.kdeplot(data = dist_train[id_samples_train], color = 'red', label = 'train', fill = True, ax = ax[1])
    sns.kdeplot(data = dist_val[id_samples_val], color = 'blue', label = 'val', fill = True, ax = ax[1])
    ax[1].legend()
    ax[1].set_title('Used')
    ax[1].set_xlabel('% drought pixels (in samples)')
    
    plt.tight_layout()
    plt.savefig(savepath + '/_distributions')
    plt.show()

def refine_probs(p_droughts_train, distr_train, p_droughts_val, distr_val, 
                 min_p_drought = 0.1, min_non_nan_content = 0.0): 
    """Converts droughts percentages lower than the min_p_drought into zero
    and filters for samples with lots of nan locations

    :param p_droughts_train: drought percentages of the training samples
    :type p_droughts_train: numpy.ndarray
    :param distr_train: mean anomalies of the input varriables of the training samples
    :type distr_train: numpy.ndarray
    :param p_droughts_val: drought percentages of the valitation samples
    :type p_droughts_val: numpy.ndarray
    :param distr_val: mean anomalies of the input varriables of the valitation samples
    :type distr_val: numpy.ndarray
    :param min_p_drought: min percent of drought content, defaults to 0.1
    :type min_p_drought: float, optional
    :param min_non_nan_content: min non nan content, defaults to 0.0
    :type min_non_nan_content: float, optional
    :return: Filtered arrays according to the conditions
    :rtype: np.ndarrays
    """
    # Convert to non drought the samples that do not have "enought" drought content
    p_droughts_train = np.where(p_droughts_train < min_p_drought, 0, p_droughts_train)
    p_droughts_val = np.where(p_droughts_val < min_p_drought, 0, p_droughts_val)
        
    # Delete samples with less than the minimum non-water content
    mask_train = (np.sum(~np.isnan(distr_train), axis = 1) > min_non_nan_content) 
    mask_val = (np.sum(~np.isnan(distr_val), axis = 1) > min_non_nan_content) 
    
    return (p_droughts_train[mask_train], distr_train[mask_train],
            p_droughts_val[mask_val], distr_val[mask_val])

def sample_adaptation(p_droughts_train, distr_train, p_droughts_val, distr_val, alpha = 0.05):
    """Modifies the ids of the samples taken such that the distribution of in-sample drought content 
    for training and validation is similar. 

    :param p_droughts_train: drought percentages of the training samples
    :type p_droughts_train: numpy.ndarray
    :param distr_train: mean anomalies of the input varriables of the training samples
    :type distr_train: numpy.ndarray
    :param p_droughts_val: drought percentages of the valitation samples
    :type p_droughts_val: numpy.ndarray
    :param distr_val: mean anomalies of the input varriables of the valitation samples
    :type distr_val: numpy.ndarray
    :param alpha: value for the independent test that checks that the two distributions 
    come from the same underlying one, defaults to 0.05
    :type alpha: float, optional
    :return: arrays of ids indicating which samples from the dataset can be taken
    :rtype: np.ndarrays
    """    
    # Get the histogram of each array of probabilities
    n_bins = int(np.sqrt(np.minimum(len(p_droughts_train), len(p_droughts_val)))) # square-root choice
    _, bin_edges = np.histogram(p_droughts_train, bins = n_bins, range = (0.0, 100.0)) # bin edges goes from zero to (length(hist)+1)
    _, _ = np.histogram(p_droughts_val, bins = n_bins, range = (0.0, 100.0)) # range is in percentages, from 0 to 100
    
    # All but the last (righthand-most) bin is half-open. The last bin, is closed. 
    # We do a small trick to prevent missing elements (if there is any that has pdrought = 100%)
    # in the selection later on. When we get the elements < bin_edges
    bin_edges[-1] += 0.1
    
    # Initialize
    id_samples_train = np.ones(np.shape(p_droughts_train)[0])
    id_samples_val = np.ones(np.shape(p_droughts_val)[0])
    
    # Remove one all the samples that do not belong 
    # to the same underlying distribution for each bin
    for idx_bin in range(n_bins):
        
        # Get the trainning and validation samples of that bin
        idxs_bin_samples_train = np.flatnonzero(np.logical_and(bin_edges[idx_bin] <= p_droughts_train, 
                                                               p_droughts_train < bin_edges[idx_bin + 1]))
        idxs_bin_samples_val = np.flatnonzero(np.logical_and(bin_edges[idx_bin] <= p_droughts_val, 
                                                             p_droughts_val < bin_edges[idx_bin + 1]))
        
        # Check combinations of a train and val sample 
        # for a non significant pvalue.
        # we can never prove the null hypothesis; 
        # all we can do is reject or fail to reject it. 
        # Here we say, all samples are assumed to belong to the same distribution but, 
        # I'll test sample by sample and in case that it is proven that a sample in train/val
        # does not share underlying common distribution with any val/train samples, I'll remove that sample 
        l_train = len(idxs_bin_samples_train)
        l_val = len(idxs_bin_samples_val)
        p_value_matrix = np.empty((l_train, l_val))
        for i, j in product(range(l_train), range(l_val)):
            
            # sample distribution
            distr_sample_train = distr_train[idxs_bin_samples_train[i]]
            distr_sample_val = distr_val[idxs_bin_samples_val[j]]
            
            # Compute the test
            p_value = ttest_ind(distr_sample_train, distr_sample_val,
                                equal_var = False, nan_policy = 'omit', 
                                alternative = 'two-sided').pvalue    
            p_value_matrix[i, j] = p_value

        # Check condition of p_value and remove samples
        p_value_matrix = (p_value_matrix < alpha)
        p_value_vector_train = p_value_matrix.prod(axis = 1)
        p_value_vector_val = p_value_matrix.prod(axis = 0)

        # Modify sample vector
        id_samples_train[idxs_bin_samples_train * p_value_vector_train] = 0
        id_samples_val[idxs_bin_samples_val * p_value_vector_val] = 0
                           
    n_samples_removed_train = len(id_samples_train) - sum(id_samples_train)
    n_samples_removed_val = len(id_samples_val) - sum(id_samples_val)
    
    print(f'--Number of samples removed train: {int(n_samples_removed_train)} of {len(id_samples_train)}')
    print(f'--Number of samples removed val: {int(n_samples_removed_val)} of {len(id_samples_val)}')
    
    assert n_samples_removed_train != len(id_samples_train), 'All samples removed in train'
    assert n_samples_removed_val != len(id_samples_val), 'All samples removed in val'
    
    return id_samples_train, id_samples_val

def get_ids(p_droughts_train, distr_train, p_droughts_val, distr_val, 
            mode = 'independent', alpha = 0.05):
    """Get ids indicating which samples from the dataset are taken

    :param p_droughts_train: drought percentages of the training samples
    :type p_droughts_train: numpy.ndarray
    :param distr_train: mean anomalies of the input varriables of the training samples
    :type distr_train: numpy.ndarray
    :param p_droughts_val: drought percentages of the valitation samples
    :type p_droughts_val: numpy.ndarray
    :param distr_val: mean anomalies of the input varriables of the valitation samples
    :type distr_val: numpy.ndarray
    :param mode: mode of modification of the sample distibution, defaults to 'independent'
    :type mode: str, optional
    :param alpha: value for the independent test that checks that the two distributions 
    come from the same underlying one, defaults to 0.05
    :type alpha: float, optional
    :return: arrays of ids indicating which samples from the dataset can be taken
    :rtype: np.ndarrays
    """    
    print(f'-Getting sample ids. Mode: "{mode}"')
    
    id_samples_train = p_droughts_train.any(axis = 1) # True for drought, False for no drought
    id_samples_val = p_droughts_val.any(axis = 1) # True for drought, False for no drought

    if mode == 'similar': 
        # kind of a domain adaptation to combate covariate shift 
        # this should be done independly to drought and non drought class
        # but we focus only on the drought classs
        # Adapt samples belonging to the drought class
        similar_train, similar_val = sample_adaptation(p_droughts_train[id_samples_train], distr_train[id_samples_train],
                                                       p_droughts_val[id_samples_val], distr_val[id_samples_val],
                                                       alpha = alpha)
        
        id_samples_train[id_samples_train] = similar_train
        id_samples_val[id_samples_val] = similar_val
        
    return id_samples_train, id_samples_val

def num_samples(period, id_samples, mult_factor = 1):
    """Defines the number of samples to take 

    :param period: period that defines the train or validation dataset
    :type period: str
    :param id_samples: arrays of ids indicating which samples from the dataset can be taken
    :type id_samples: np.ndarrays
    :param mult_factor: multiplicative factor to increase/decrease the number of samples. 
    Values bigger than 1 will add non drought samples to the dataloader, defaults to 1
    :type mult_factor: int, optional
    :return: number of samples to use
    :rtype: int
    """    
    # Number of drought samples
    n_dsamples = int(sum(id_samples))
    assert n_dsamples != 0, 'No drought samples in the data'

    # Total number of samples
    NUM = int(np.ceil(n_dsamples * mult_factor))
    assert NUM <= len(id_samples), 'NUM samples > nº samples in the dataset'
            
    print(f'-nº drought samples in {period}: {n_dsamples} -> nº samples used each epoch: {NUM}')
    
    return NUM

def define_sampler(period, NUM, id_samples, fixed_samples = False):
    """The indices in the dataset are ordered from higher to lower percentages of drought pixels
    to modify the number of samples to work with, we can use SubsetRandomSampler o WeightedRandomSampler. 
    To allow randomness in the selection we use the latter.

    :param period: period that defines the train or validation dataset
    :type period: str
    :param NUM: number of samples to use
    :type NUM: int
    :param id_samples: arrays of ids indicating which samples from the dataset can be taken
    :type id_samples: np.array
    :param fixed_samples: fixes the samples seen each epoch of training, defaults to False
    :type fixed_samples: bool, optional
    :return: sampler for the corresponding dataset
    :rtype: torch.utils.data.sampler
    """    
    # Weights of the samples
    pdefault = 1e-7 # default probability. Small but not zero 
    weights = np.ones(len(id_samples)) * pdefault 
    weights[id_samples] = 1 - pdefault # probability for drought samples (True == 1)
    
    # Fix samples
    # Turns the probability of id_sample > NUM to zero. Only the N first elements are kept.
    # Avoids selecting non-previously seen samples between epochs. This can happen if some
    # elements in the pool have the same probability and not all are taken in each epoch, i.e.:
    # a) we use non-drought samples (and default_prob != 0), or
    # b) max_samples is not None and NUM < number of drought samples
    # This is the situation when training and having the capping 
    # in the number of samples between train and val  
    if fixed_samples: 
        
        print(f'-Fixed samples for {period}. Same for all epochs.')
        weights[NUM:] = 0
    
    # Define the sampler
    drought_sampler = sampler.WeightedRandomSampler(weights, num_samples = NUM, replacement = False) 
    # the sampler is random also for validation, but because we only look at the metrics 
    # for all samples in validation, to have this randomness does not matter. We see all the validation dataset each time
    
    return drought_sampler

def define_dataloaders(GDIS, experiment_config, dataset_config):
    """Given the experimental and dataset configs, 
    returns the dataloaders for training, validation and testing

    :param GDIS: GDIS dataset 
    :type GDIS: torch.utils.data.Dataset
    :param experiment_config: experimental config
    :type experiment_config: dict
    :param dataset_config: dataset config
    :type dataset_config: dict
    :return: dataloaders for train, val and test
    :rtype: torch.utils.data.DataLoader
    """
    # Predefine the dataloaders
    loader_train = None
    loader_val = None
    loader_test = None
    
    if experiment_config['Arguments']['doFit']:
        
        # datasets
        data_train = GDIS(dataset_config, period = 'train') 
        data_val = GDIS(dataset_config, period = 'val')
        
        # refine samples
        print('\nRefine samples...')
        # Change the probabilities by filtering for a minimum of drought content 
        # Also delete samples over the water 
        p_droughts_train, distr_train, p_droughts_val, distr_val = refine_probs(
                                                                        data_train.samples_info['p_drought_idxs'].numpy(), 
                                                                        data_train.samples_info['distr_idxs'].numpy(),
                                                                        data_val.samples_info['p_drought_idxs'].numpy(),
                                                                        data_val.samples_info['distr_idxs'].numpy(),
                                                                        min_p_drought = dataset_config['GDIS']['min_p_drought'],
                                                                        min_non_nan_content = dataset_config['GDIS']['min_non_nan_content'])

        # Get ids. The number of drought pixels in each sample is different for train and val splits
        # this incurs in different speeds for the change of the metrics (AUROC) in training and validation
        # we refine the samples so the distributions are similar with mode = 'similar'. Default is 'independent'
        id_samples_train, id_samples_val = get_ids(p_droughts_train, distr_train,
                                                   p_droughts_val, distr_val,
                                                   mode = dataset_config['GDIS']['distr_mode'],
                                                   alpha = dataset_config['GDIS']['alpha'])
        
        # Plot the distributions of the drought samples
        visualize_dist(p_droughts_train, 
                       p_droughts_val, 
                       id_samples_train, id_samples_val, 
                       experiment_config['run_path'])
        
        # Get the number of samples 
        NUM_train = num_samples('train', id_samples_train, mult_factor = dataset_config['GDIS']['mf_sampler_train'])
        NUM_val = num_samples('val', id_samples_val, mult_factor = dataset_config['GDIS']['mf_sampler_val'])

        # samplers
        sampler_train = define_sampler('train', NUM_train, id_samples_train, 
                                        fixed_samples = dataset_config['GDIS']['fixed_samples_train'])
                                        
        sampler_val = define_sampler('val', NUM_val, id_samples_val, 
                                     fixed_samples = dataset_config['GDIS']['fixed_samples_val'])
        
        # dataloaders
        loader_train = DataLoader(data_train, sampler = sampler_train,
                                  batch_size = dataset_config['GDIS']['batch_size'],
                                  num_workers = dataset_config['GDIS']['num_workers'])
                                  
        loader_val = DataLoader(data_val, sampler = sampler_val,
                                batch_size = dataset_config['GDIS']['batch_size'],
                                num_workers = dataset_config['GDIS']['num_workers'])

        # save the len to the dataset config
        dataset_config['len_loader_train'] = len(loader_train)
    
    if (experiment_config['Arguments']['doTest'] or 
        experiment_config['Arguments']['doInference']):
        
        # dataset
        data_test = GDIS(dataset_config, period = 'test')
        
        # dataloader (using the sequential sampler by default)
        # batch_size = 1, if more than one there is redundant computation and not reliable results
        loader_test = DataLoader(data_test, num_workers = 0) 
    
    return loader_train, loader_val, loader_test, dataset_config
