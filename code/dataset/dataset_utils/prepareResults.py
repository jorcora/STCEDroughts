#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# GENERAL
import os

# NUMPY 
import numpy as np

# XARRAY
import xarray as xr

# PICKLE
import pickle

# DASK
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

# UTILS
from utils import space_time_slice

class PrepRES:
    def __init__(self, dataset_config, experiment_config):
        self.dataset_config = dataset_config['GDIS']
        self.dataset_config_indices = dataset_config['DInd']
        self.dataset_config_context = dataset_config['HWSD']
        self.experiment_config = experiment_config
        
        # Correction for the model's lost border and resolution correspondence
        size_in = list(eval(self.dataset_config['input_size']))
        size_out = list(eval(self.dataset_config['output_size']))
        self.lost_border = int((size_in[0] - size_out[0])/2)
        idt = self.dataset_config['idt']
        self.time_border = self.dataset_config['idt_ref_in'][idt]
        
        print("\nPrepare data...")
        
        #######################  Loading Data  #########################    
        
        print('-Loading labels')
        # >>> Labels for droughts
        self.labels = xr.open_zarr(os.path.join(self.dataset_config['root'], self.dataset_config['labels_file']), consolidated = False).chunk({'time': 1})
        self.labels = self.labels.transpose("time", "lat", "lon")
        self.labels = self.labels['mask'] > 0 # to convert into a bolean mask that treats all labels the same
        self.labels = self.labels * 1.0 # to make sure that the interpolation works I turn the bolean values into real ones
        
        print('-Loading indices')
        # >>> Load drought indexes
        self.indices = xr.open_zarr(os.path.join(self.dataset_config_indices['root'], self.dataset_config_indices['indices_file']), consolidated = False)
        self.indices = self.indices[np.array(self.dataset_config_indices['features'])[self.dataset_config_indices['features_selected']]]
        self.indices = self.indices.transpose('time', 'lat', 'lon').chunk({'time': 1})
        
        print('-Loading contextual masks')
        # >>> Land use context information 
        lu_context = xr.open_zarr(os.path.join(self.dataset_config_context['root'], self.dataset_config_context['context_file']), consolidated = False)
        lu_context = lu_context[np.array(self.dataset_config_context['features'])[self.dataset_config_context['features_selected']]]
        lu_context = lu_context.to_array(dim = 'var').argmax(dim ='var') # Dominant land use. Values are the indices indicating the vars. in the config list
        lu_context = lu_context.transpose('lat', 'lon')
        lu_context = lu_context.rename('LU_types')
        lu_context = xr.where(lu_context.isin([0,1,2]), x = 0, y = lu_context)
        lu_context = xr.where(lu_context.isin([3,4]), x = 1, y = lu_context)
        lu_context = xr.where(lu_context.isin([5,6]), x = 2, y = lu_context)

        # >>> water conditions context information 
        # this is going to be described by soil root moisture levels
        # ROOT MOISTURE
        rm_context = xr.open_zarr(os.path.join(self.dataset_config['root'], self.dataset_config['data_file'], 'root_moisture.zarr'), consolidated = False)
        rm_context = rm_context.transpose('time', 'lat', 'lon').chunk({'time': 1})
        rm_context = rm_context['root_moisture'].rename('RM_levels')
        for class_id, n_level in enumerate(np.array([0.00, 0.10, 0.30, 0.50, 0.70, 1.00])): # levels
            rm_context = xr.where(rm_context <= n_level, x = class_id + 1, y = rm_context) # start from 1 to avoid complications
        rm_context -= 1 
        # Root moisture goes from 0.0 (wilting point - completely dry) to 1.0 (saturation - fully wet) 
        
        # >>> seasonal context information
        szn_context = xr.zeros_like(self.labels).rename('SZN_periods')
        for class_id, szn_interval in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            szn_context = szn_context.where(~szn_context['time.season'].isin(szn_interval), other = class_id)
        
        # Combine the masks
        self.context =  xr.merge([lu_context, rm_context, szn_context],
                                 join = 'outer').transpose('time', 'lat', 'lon').chunk({'time': 1})
        
        ######################  Miscellaneous  ######################### 
        
        # >>> Indices
        # Identify infinities and replace with nans
        self.indices = self.indices.where(np.isfinite(self.indices), other = np.nan)
        
        # All the indices are defined with negative values corresponding to more dry conditions
        # But the provided CDI comes with the reversed sign:
        if 'CDI' in np.array(self.dataset_config_indices['features'])[self.dataset_config_indices['features_selected']]:
            self.indices['CDI'] = -self.indices['CDI']
        
        # Flip the sign of the indices so bigger positive values correspond to higher/more intense 
        # classes of droughts. Same criterion as the scores of the model.
        self.indices = -self.indices
        
        ######################  Preprocessing  #########################
        
        # >>> Spatial slices
        self.indices = space_time_slice(self.dataset_config, self.indices, objects_to_slice = ('time', 'space'))
        self.labels = space_time_slice(self.dataset_config, self.labels, objects_to_slice = ('time', 'space'))
        self.context = space_time_slice(self.dataset_config, self.context, objects_to_slice = ('time', 'space'))
        
    def _get_results_indices_(self, model_type = 'alpha', labels_type = 'labels'): # spx_labels
        """
        """
        print(f'-Loading model results. Type of model: {model_type}. Type of labels: {labels_type}') 
        
        # Get results of the defined or all folds
        if self.experiment_config['Arguments']['fold_id'] == 'FA':
            print('--Loading all folds and concatenating')
            
            # Test periods: 
            # F1 -> ["'January-2011','December-2015'"]
            # F3 -> ["'January-2009','December-2010'"]
            # F5 -> ["'January-2003','December-2008'"]
            # For a chronological order we set: [F5, F3 and F1]
            # Xarray doesn't allow an empy datarray or dataset, so we 
            # initialize with the results of the first period.
            acc_model, acc_indices = self._adapt_results_indices_('F5', model_type, labels_type)
            for fold in ['F3', 'F1']:
                
                # Get results
                fold_model, fold_indices = self._adapt_results_indices_(fold, model_type, labels_type)
                
                # Accumulate results
                acc_model = xr.concat([acc_model, fold_model], dim = 'time')
                acc_indices = xr.concat([acc_indices, fold_indices], dim = 'time')
        
        else:
            print('--Loading fold', self.experiment_config['Arguments']['fold_id'])
            fold = self.experiment_config['Arguments']['fold_id']
            
            # Get results
            acc_model, acc_indices = self._adapt_results_indices_(fold, model_type, labels_type)
            
        # Interpolate and convert to numpy dictionaries
        interp_model, interp_indices = self._extract_interpolate_(acc_model, acc_indices)
        
        return interp_model, interp_indices
    
    def _adapt_results_indices_(self, fold, model_type = 'alpha', labels_type = 'labels'):
        """
        """
        # Temporal selection according to the fold
        indices2compare = space_time_slice(self.dataset_config['folds'][fold], self.indices, period = 'test', objects_to_slice = ('period'))
        labels_dataset = space_time_slice(self.dataset_config['folds'][fold], self.labels, period = 'test', objects_to_slice = ('period'))
        context_dataset = space_time_slice(self.dataset_config['folds'][fold], self.context, period = 'test', objects_to_slice = ('period'))
        
        # Load model
        with open(self.experiment_config[f'pretrained_model_path_{fold}_{model_type}'], "rb") as f:
            results_maps = pickle.load(f)

        # Create model dataset from the results maps
        self.id_events = np.shape(results_maps['map_events'])[1]
        self.len_time = np.shape(results_maps['y_hat'])[0]
        model2compare = xr.Dataset(
            data_vars = dict(y_hat = (["time", "lat", "lon"], np.array(results_maps['y_hat'])),
                             masks = (["time", "lat", "lon"], np.array(results_maps['masks'])),
                             labels = (["time", "lat", "lon"], np.array(results_maps[f'{labels_type}'])),
                             map_events = (["time", "nev", "lat", "lon"], np.array(results_maps['map_events']))), # this are real values, not bolean ones, They work better for interpolating (scipy application)
            coords = dict(nev = range(self.id_events),
                          lon = labels_dataset.lon[self.lost_border:], 
                          lat = labels_dataset.lat[self.lost_border:], 
                          time = labels_dataset.time[self.time_border:self.time_border + self.len_time])) # Temporal reference of the labels, consider how the ouputs have missed positions
        
        # Adapt the contextual information to the model sizes
        context_dataset = context_dataset.isel(lat = np.arange(self.lost_border, context_dataset.dims['lat']),
                                               lon = np.arange(self.lost_border, context_dataset.dims['lon']),
                                               time = np.arange(self.time_border, self.time_border + self.len_time))
        
        # Merge to the model2compare variable (so all variables with the same res are together)
        model2compare = xr.merge([model2compare, context_dataset])
        
        # Correct indices for the lost spatial border of the model
        indices2compare = indices2compare.isel(lat = np.arange(self.lost_border, indices2compare.dims['lat']),
                                               lon = np.arange(self.lost_border, indices2compare.dims['lon']))
            
        return model2compare, indices2compare
    
    def _extract_interpolate_(self, acc_model, acc_indices):
        """
        """
        # Predefine and extract the variables
        dict_model2compare = {}
        for key in list(acc_model.keys()):
            dict_model2compare[key] = acc_model[key].values
        
        dict_indices2compare = {}
        for key in list(acc_indices.keys()):
            dict_indices2compare[key] = acc_indices[key].values
        
        # Get the time of each object
        time_mod = acc_model.time.values
        time_ind = acc_indices.time.values
        
        # Interpolate
        if self.experiment_config['Arguments']['ref_timescale'] == 'indices': 
            
            self.time_ref = time_ind
            dict_model2compare = self._near_interpolate_(time_mod, dict_model2compare, fill_value = 0)
        
        elif self.experiment_config['Arguments']['ref_timescale'] == 'model':
            
            self.time_ref = time_mod
            dict_indices2compare = self._near_interpolate_(time_ind, dict_indices2compare, fill_value = np.nan)
            
        return dict_model2compare, dict_indices2compare
    
    def _near_interpolate_(self, time_obj, dict_var, fill_value = np.nan, 
                           skip_nanIm = True, nonans = np.ones(1), max_temporal_jump = 9999): 
        # if setting skip_nanIm = False, the common mask can remove all positions
        """
        Interpolate the dataset to fit the temporal scale of the reference dataset 
        Take care that the indices dataset is created by agregatting all indices, 
        into a common temporal range and this is accomplished by filling empty time steps with nan images. 
        We have to remove them prior to interpolation with the model
        Depending if we are interpolating indices or the model results, the value to fill the variables
        is different. If interpolating indices it is nans to be catched by the nan_mask of the indices; 
        if it is the models results, it has to be zero to be interpreted as valid values of the mask. 
        """
        print('Matching the timescales of model and indices')
        print('Reference timescale:', self.experiment_config['Arguments']['ref_timescale'])
        
        # Loop for every variable
        int_dict_var = {}
        for key in list(dict_var.keys()):
            print('-Interpolating', key)
            
            # Create the dummy variable
            dummy_var = np.ones((len(self.time_ref),) + np.shape(dict_var[key])[1:]) * fill_value 
            
            if skip_nanIm:
                # Creates a mask to avoid empty images
                nonans = np.logical_not(np.isnan(dict_var[key]).all(axis = (-2, -1)))
            
            for i, id_time in enumerate(self.time_ref):    
                
                # Loop time steps looking for the most recent past instance that is not full of nans.
                # To control the temporal jump, we define the variable max_temporal_jump. 
                tmp = (time_obj - id_time) / np.timedelta64(1, 'D')
                tmp = np.logical_and(tmp <= 0, -max_temporal_jump <= tmp) * nonans
                tmp = np.where(tmp)[0]
                
                # Check if there is a past instance and fill the values 
                if tmp.any():
            
                    past_idx = tmp[-1]
                    dummy_var[i] = np.copy(dict_var[key][past_idx])
            
            int_dict_var[key] = np.copy(dummy_var)
        
        return int_dict_var
