#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi cortés-andrés
"""

# NUMPY
import numpy as np

# PYTORCH
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# PYTORCH LIGHTNING
import pytorch_lightning as pl

# AM
from .am import LAM

class SHIFT4L(pl.LightningModule):
  def __init__(self, model_config):
    super().__init__()
    print('Model:', model_config['arch']['name'])
      
    # Save hyperparameters
    self.save_hyperparameters()
      
    # Configs
    self.model_config = model_config

    # Params
    conv_params = {'kernel_size': eval(self.model_config['arch']['conv_kernel_size']), 'stride': (1,1,1)}
    last_conv_params = {'kernel_size': eval(self.model_config['arch']['last_conv_kernel_size']), 'stride': (1,1,1)}
    pool_kernel_size = eval(self.model_config['arch']['pool_kernel_size'])

    # batch norm params
    bnorm_params = {'momentum': self.model_config['arch']['momentum'], 'track_running_stats': True}
      
    # Encoder
    self.encoder_conv_1 = nn.Conv3d(self.model_config['arch']['in_channels_climate'], 
                                    self.model_config['arch']['encoder_1'], 
                                    **conv_params)
      
    self.encoder_conv_2 = nn.Conv3d(self.model_config['arch']['encoder_1'], 
                                    self.model_config['arch']['encoder_2'], 
                                    **conv_params)
    
    self.encoder_conv_3 = nn.Conv3d(self.model_config['arch']['encoder_2'],
                                    self.model_config['arch']['encoder_3'],
                                    **conv_params)

    self.encoder_conv_4 = nn.Conv3d(self.model_config['arch']['encoder_3'],
                                    self.model_config['arch']['encoder_4'],
                                    **conv_params)
      
    #  Normalization layers encoder
    self.encoder_bn_1 = nn.BatchNorm3d(self.model_config['arch']['encoder_1'], **bnorm_params)
    self.encoder_bn_2 = nn.BatchNorm3d(self.model_config['arch']['encoder_2'], **bnorm_params)
    self.encoder_bn_3 = nn.BatchNorm3d(self.model_config['arch']['encoder_3'], **bnorm_params)
    self.encoder_bn_4 = nn.BatchNorm3d(self.model_config['arch']['encoder_4'], **bnorm_params)
    
    # LOCATION ATTENTION MECHANISM
    #self.LAM = LAM(self.model_config)

    # Decoder
    self.decoder_conv_4 = nn.Conv3d(self.model_config['arch']['decoder_4'], 
                                    self.model_config['arch']['decoder_3'], 
                                    **conv_params)

    self.decoder_conv_3 = nn.Conv3d(self.model_config['arch']['decoder_3'], 
                                    self.model_config['arch']['decoder_2'], 
                                    **conv_params)
    
    self.decoder_conv_2 = nn.Conv3d(self.model_config['arch']['decoder_2'],
                                    self.model_config['arch']['decoder_1'],
                                    **conv_params)
    
    self.decoder_conv_1 = nn.Conv3d(self.model_config['arch']['decoder_1'], 
                                    self.model_config['arch']['in_channels_climate'], 
                                    **conv_params)
    
    # Normalization layers decoder
    self.decoder_bn_4 = nn.BatchNorm3d(self.model_config['arch']['encoder_3'], **bnorm_params)
    self.decoder_bn_3 = nn.BatchNorm3d(self.model_config['arch']['encoder_2'], **bnorm_params)
    self.decoder_bn_2 = nn.BatchNorm3d(self.model_config['arch']['encoder_1'], **bnorm_params)
    self.decoder_bn_1 = nn.BatchNorm3d(self.model_config['arch']['in_channels_climate'], **bnorm_params)
      
    # Output
    self.out_conv = nn.Conv3d(self.model_config['arch']['in_channels_climate'], 
                              self.model_config['arch']['num_classes'], 
                              **last_conv_params)
      
    # Initializate the bias of inner layers and the output layer
    # Encoder
    torch.nn.init.constant_(self.encoder_conv_1.bias, 0)
    torch.nn.init.constant_(self.encoder_conv_2.bias, 0)
    torch.nn.init.constant_(self.encoder_conv_3.bias, 0)
    torch.nn.init.constant_(self.encoder_conv_4.bias, 0)
    
    # Decoder
    torch.nn.init.constant_(self.decoder_conv_4.bias, 0)
    torch.nn.init.constant_(self.decoder_conv_3.bias, 0)
    torch.nn.init.constant_(self.decoder_conv_2.bias, 0)
    torch.nn.init.constant_(self.decoder_conv_1.bias, 0)
    
    # Output
    bias = -np.log((1 - self.model_config['arch']['PI'])/self.model_config['arch']['PI']) 
    torch.nn.init.constant_(self.out_conv.bias, bias)
      
    # Activation
    self.activation  = nn.LeakyReLU(self.model_config['arch']['slope'])
    
    # Dropout
    self.dropout = nn.Dropout3d(p = self.model_config['arch']['dropout'])
    
    # Pooling and upsampling
    self._2d_pool = nn.MaxPool3d(kernel_size = pool_kernel_size, stride = pool_kernel_size)
    self._2d_upsample = nn.Upsample(scale_factor = pool_kernel_size, mode = 'trilinear', align_corners = False)
      
  def forward(self, x, step_samples): 
        
    # ENCODER                                                           
    #1
    skip1 = x                                                         
    x = self.encoder_conv_1(x)                                          
    x = self.encoder_bn_1(x)                                                                     
    x = self.activation(x)
    x = self.dropout(x)                                                                                                        
    x = self._2d_pool(x)  
    
    #2
    skip2 = x
    x = self.encoder_conv_2(x)                                          
    x = self.encoder_bn_2(x)                                                                    
    x = self.activation(x)
    x = self.dropout(x)                                                  
    x = self._2d_pool(x)
    
    #3
    skip3 = x
    x = self.encoder_conv_3(x)                                          
    x = self.encoder_bn_3(x)                                                                      
    x = self.activation(x)
    x = self.dropout(x)                                                   
    x = self._2d_pool(x) 

    #4
    skip4 = x
    x = self.encoder_conv_4(x)                                          
    x = self.encoder_bn_4(x)                                                                      
    x = self.activation(x)
    x = self.dropout(x)                                                   
    x = self._2d_pool(x) 

    #4
    x = self._2d_upsample(x)
    x = self.decoder_conv_4(x)                                          
    x = self.decoder_bn_4(x)
    time_diff = int(skip4.size(2) - x.size(2))
    x = x + TF.center_crop(skip4, x.size()[-2:])[:,:,time_diff:,:,:] 
    x = self.activation(x)
    x = self.dropout(x)  
    
    #3
    x = self._2d_upsample(x)
    x = self.decoder_conv_3(x)                                          
    x = self.decoder_bn_3(x)
    time_diff = int(skip3.size(2) - x.size(2))
    x = x + TF.center_crop(skip3, x.size()[-2:])[:,:,time_diff:,:,:] 
    x = self.activation(x)
    x = self.dropout(x)                                                   
    
    #2
    x = self._2d_upsample(x)
    x = self.decoder_conv_2(x)                                          
    x = self.decoder_bn_2(x)
    time_diff = int(skip2.size(2) - x.size(2))
    x = x + TF.center_crop(skip2, x.size()[-2:])[:,:,time_diff:,:,:] 
    x = self.activation(x)
    x = self.dropout(x)                                                 
    
    #1
    x = self._2d_upsample(x)
    x = self.decoder_conv_1(x)                                          
    x = self.decoder_bn_1(x)
    time_diff = int(skip1.size(2) - x.size(2))
    x = x + TF.center_crop(skip1, x.size()[-2:])[:,:,time_diff:,:,:] 
    x = self.activation(x)
    x = self.dropout(x)                                                                 
          
    #Output
    x = self.out_conv(x)                                            
    
    return x


class LSTM(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        print('Model:', model_config['arch']['name'])

        # Save hyperparameters
        self.save_hyperparameters()
        
        # Configs
        self.model_config = model_config
        
        # LSTM definition
        self.lstm = nn.LSTM(self.model_config['arch']['in_channels_climate'], # The number of expected features in the input x
                            self.model_config['arch']['hidden_dims'], # The number of features in the hidden state h
                            self.model_config['arch']['n_layers'], # Number of recurrent layers (stacked LSTMs)
                            batch_first = True) # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature)
        
        # Readout layer
        self.fc = nn.Linear(self.model_config['arch']['hidden_dims'], self.model_config['arch']['num_classes'])

    def forward(self, x, step_samples):
        
        # Adapt the input 
        lat_lon_sizes = x.size(-1)
        x = torch.reshape(x, (x.size(0), x.size(1), x.size(2), -1))
        x = x.permute(0,2,1,3)
        
        # Compute for each spatial location
        out = torch.ones((x.size(0), 1, x.size(1), x.size(3))) * -999
        for n_loc in np.arange(0, x.size(-1), step_samples): 
 
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.model_config['arch']['n_layers'], x.size(0), 
                             self.model_config['arch']['hidden_dims']).to(x.device).requires_grad_()
        
            # Initialize cell state
            c0 = torch.zeros(self.model_config['arch']['n_layers'], x.size(0), 
                             self.model_config['arch']['hidden_dims']).to(x.device).requires_grad_()
        
            # 10 time steps but tremember that we retain only the last two 
            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            tmp_out, _ = self.lstm(x[:,:,:,n_loc], (h0.detach(), c0.detach()))
            
            # Index hidden state of last time step
            tmp_out = self.fc(tmp_out)
            
            # Change order and add in the corresponding position
            tmp_out = torch.permute(tmp_out, (0,2,1))
            out[:,:,:,n_loc] = tmp_out

        # Restore original size
        out = torch.reshape(out, (out.size(0), out.size(1), out.size(2), 
                                  lat_lon_sizes, lat_lon_sizes))
        
        return out
