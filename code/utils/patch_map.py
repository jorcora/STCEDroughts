#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi cortés-andrés
"""

# NUMPY
import numpy as np

# TORCH
import torch
import torch.nn.functional  as F

# ITERTOOLS
from itertools import product  

class M2P_P2M:
    """Class to take patches from the complete map 
    and reconstruct the complete map from patches
    """    
    def __init__(self, variable, input_size, output_size, stride = 64, generate_map = True): 
        """_summary_

        :param variable: input variable
        :type variable: torch.Tensor
        :param input_size: input size of the patch
        :type input_size: tuple of ints
        :param output_size: output size of the patch
        :type output_size: tuple of ints
        :param stride: step between samples, it should be the size 
        of the ouput patch for correct reconstruction, defaults to 64
        :type stride: int, optional
        :param generate_map: defines it the map has to be reconstructed or not, 
        saves memory if not, defaults to True
        :type generate_map: bool, optional
        """        
        # Define the stride
        self.stride = stride
        
        # Input and output size
        self.input_size = input_size[0]
        self.output_size = output_size[0]
        time_out = output_size[2]
        
        # Diference between input and output sizes after going through the model
        self.lost_border = int((self.input_size - self.output_size)/2)
        
        # Variable
        self.variable = variable
        
        # Lengths of the complete map
        self.len_lat = self.variable.size(-2)
        self.len_lon = self.variable.size(-1)
        
        # Get the number of patches in each spatial dimension
        self.n_lat_patch = int(np.ceil(self.len_lat / self.stride))
        self.n_lon_patch = int(np.ceil(self.len_lon / self.stride))
        
        # Padding 
        pad_size = (0, self.input_size, 0, self.input_size) # pads only the last two dimensions
        self.variable = F.pad(self.variable, pad_size, mode = 'constant', value = 0) 
        
        # Create an empty tensor to acomodate all patches for posterior reconstruction
        # Note that the specific spatial size doesn't matter. It just has to be big enought.
        # Temporal dimension has to be the one for the ouput
        # Same for the variable dimension (it has to be one)
        if generate_map:
        
            self.reconstructed_map = torch.zeros_like(self.variable)
            self.reconstructed_map = self.reconstructed_map[:, :1, :time_out, :, :]
        
        # Create iterable list. The first element identifies the lat, the second the lon
        self.idxs = list(product(np.arange(self.n_lat_patch), np.arange(self.n_lon_patch)))
        
    def get_patch(self, idx):
        """Get patch

        :param idx: idx of the patch
        :type idx: tuple of ints, lat and lon
        :return: cropped input variable (patch at the input)
        :rtype: torch.Tensor
        """        
        lat = idx[0] * self.stride
        lon = idx[1] * self.stride
        patch = self.variable[:, :, :, 
                              lat:lat + self.input_size, 
                              lon:lon + self.input_size]
        
        return patch
    
    def reconstruct_map(self, patch, idx):
        """Reconstruct map

        :param patch: patch at the ouput
        :type patch: torch.Tensor
        :param idx: idx of the patch
        :type idx: tuple of ints, lat and lon
        """        
        lat = idx[0] * self.stride + self.lost_border
        lon = idx[1] * self.stride + self.lost_border
        self.reconstructed_map[:, :, :, 
                               lat:lat + self.output_size, 
                               lon:lon + self.output_size] = patch
        
    def get_reconstructed_map(self):
        """Gets the reconstructed map

        :return: reconstructed variable for the full map
        :rtype: torch.Tensor
        """        
        # Correct for the added padding 
        self.reconstructed_map = self.reconstructed_map[:, :, :,
                                                        :self.len_lat,
                                                        :self.len_lon]
        
        # Remove the artificial border added in the left and upper side 
        self.reconstructed_map = self.reconstructed_map[:, :, :, 
                                                        self.lost_border:,
                                                        self.lost_border:]
        
        return self.reconstructed_map
    
