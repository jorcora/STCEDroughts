#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY 
import numpy as np

# XARRAY
import xarray as xr

# TORCHVISION
import torchvision.transforms as T

def space_time_slice(config, dataset, period = None, objects_to_slice = ('time', 'period', 'space')):
    """Slices time and space dimensions according to config info

    :param config: dataset config
    :type config: dict
    :param dataset: input dataset class
    :type dataset: torch.utils.data.Dataset
    :param period: time period, defaults to None
    :type period: str, optional
    :param objects_to_slice: what to slice, defaults to ('time', 'period', 'space')
    :type objects_to_slice: tuple, optional
    :return: sliced dataset 
    :rtype: torch.utils.data.Dataset
    """
    sliced_dataset  = dataset.copy(deep = True)
    if 'time' in objects_to_slice:
        
        # Temporal slice
        start, end = eval(config['time_slice'])
        sliced_dataset = sliced_dataset.sel(time = slice(start, end))
        
    if 'period' in objects_to_slice:
        
        # Train or Validation or Test period
        data_aux = []
        for interval in np.arange(len(config[f'{period}_slice'])):
            start, end = eval(config[f'{period}_slice'][interval])
            data_aux.append(sliced_dataset.sel(time = slice(start, end)))
            
        sliced_dataset = xr.concat(data_aux, dim = 'time').chunk({'time': 1})
    
    if 'space' in objects_to_slice:
        
        # Spatial slice
        start, end = eval(config['lat_slice'])
        sliced_dataset = sliced_dataset.sel(lat = slice(start, end))
        start, end = eval(config['lon_slice'])
        sliced_dataset = sliced_dataset.sel(lon = slice(start, end))
        
    return sliced_dataset

def crop_variable(config, variable_in, spatial_crop = False, temporal_crop = False):
    """Crops the variable_in according to the config

    :param config: dataset config
    :type config: dict
    :param variable_in: input tensor to slice
    :type variable_in: torch.Tensor
    :param spatial_crop: activates crop over space dimensions, defaults to False
    :type spatial_crop: bool, optional
    :param temporal_crop: activates crop over time dimension, defaults to False
    :type temporal_crop: bool, optional
    :return: cropped variable
    :rtype: torch.Tensor
    """    
    ref_size = list(eval(config['output_size'])) # (lat, lon, time)
    in_size = list(variable_in.size()) # (batch, variable, time, lat, lon)
    if spatial_crop:
        if ref_size[:2] != in_size[-2:]:
            transforms = T.CenterCrop(ref_size[:2])
            variable_in = transforms(variable_in)
      
    if temporal_crop:
        if ref_size[-1] != in_size[2]:
            variable_in = variable_in[:, :, config['idt_ref_in']] 

    return variable_in
