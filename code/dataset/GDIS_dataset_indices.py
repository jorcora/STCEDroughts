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

# TORCH
import torch
import torch.utils.data 

# TORCHVISION
import torchvision.transforms as T

# DASK
import dask
dask.config.set(scheduler='synchronous')
dask.config.set({"array.slicing.split_large_chunks": True})

# SKIMAGE
from skimage.measure import label, regionprops

# ITERTOOLS
from itertools import product

# SKIMAGE
from skimage.morphology import binary_dilation

# UTILS
from utils import space_time_slice, crop_variable
from .dataset_utils import get_3Diou

# MATPLOTLIB 
import matplotlib.pyplot as plt

class GDIS(torch.utils.data.Dataset):
    """GDIS dataset 

    :param config: dataset configuration file
    :type config: dict
    :param period: split for cross-valiation  
    :type period: str
    """
    def __init__(self, config, period):
        """Constructor method
        """
        self.period = period
        self.config = config['GDIS']
        self.config_context = config['HWSD']
        self.config_indices = config['DInd']
        
        print("\nConstructing GDIS {}...".format(self.period))
        
        #######################  Loading Data  #########################
        
        print('-Loading ESDC')
        # >>> ESDC data
        # extract the data of each variable and append them. It takes all the data in the variables 
        # and then it selects the ones we are using
        data = []
        for feature in os.listdir(os.path.join(self.config['root'], self.config['data_file'])):
            data.append(xr.open_zarr(os.path.join(self.config['root'], self.config['data_file'], feature), consolidated = False))
        self.data = xr.merge(data).chunk({'time': 1})
        self.data = self.data[np.array(self.config['features'])[self.config['features_selected']]]
        self.data = self.data.transpose('time', 'lat', 'lon')
        
        print('-Loading labels')
        # >>> Labels for droughts
        self.labels = xr.open_zarr(os.path.join(self.config['root'], self.config['labels_file']), consolidated = False).chunk({'time': 1})
        self.labels = self.labels.transpose("time", "lat", "lon")
        self.labels = self.labels['mask']
        
        print('-Loading HWSD')
        # >>> Context information 
        self.context = xr.open_zarr(os.path.join(self.config_context['root'], self.config_context['context_file']), consolidated = False)
        self.context = self.context[np.array(self.config_context['features'])[self.config_context['features_selected']]]
        self.context = self.context.transpose('lat', 'lon')
        
        print('-Loading indices')
        # >>> Load drought indexes
        self.indices = xr.open_zarr(os.path.join(self.config_indices['root'], self.config_indices['indices_file']), consolidated = False)
        self.indices = self.indices[np.array(self.config_indices['features'])[self.config_indices['features_selected']]]
        self.indices = self.indices.transpose('time', 'lat', 'lon').chunk({'time': 1})
        
        ######################  Miscellaneous  #########################
        
        # >>> Indices
        # Identify infinities and replace with nans
        self.indices = self.indices.where(np.isfinite(self.indices), other = np.nan)
        
        # Flip the sign for CDI, as it goes in reverse order
        if 'CDI' in np.array(self.config_indices['features'])[self.config_indices['features_selected']]: 
            self.indices['CDI'] = -self.indices['CDI']
        
        # Flip the sign of the indices so bigger positive values correspond to higher/more intense 
        # classes of droughts. Same criterion as the scores of the model.
        self.indices = -self.indices
        
        # >>> ESDC
        # Be careful with gross_primary_productivity!
        if 'gross_primary_productivity' in np.array(self.config['features'])[self.config['features_selected']]:
            self.data['gross_primary_productivity'] = \
                self.data['gross_primary_productivity'].where(self.data['gross_primary_productivity'] != -9999.0, other = np.nan)
        
        ######################  Preprocessing  #########################
        
        print('-Preprocessing')

        # Data for climatology
        self.data_for_clim = self.data.copy(deep = True)
        self.labels_for_clim = self.labels.copy(deep = True)
        
        # >>> Temporal slices and definition of the period of the set (train, val or test)
        self.data = space_time_slice(self.config, self.data, period = self.period, objects_to_slice = ('time', 'period'))
        self.labels = space_time_slice(self.config, self.labels, period = self.period, objects_to_slice = ('time', 'period'))
        self.indices = space_time_slice(self.config, self.indices, period = self.period, objects_to_slice = ('time', 'period'))
        
        # >>> Climatology removal
        self.config['climatology_mean_root'] = os.path.join(self.config['root'], self.config['fdata'], self.config['climatology_mean_file'])
        self.config['climatology_std_root'] =  os.path.join(self.config['root'], self.config['fdata'], self.config['climatology_std_file'])
        
        # Compute climatology
        if (self.period == 'train' 
            and not (os.path.isdir(self.config['climatology_mean_root']) 
                     or os.path.isdir(self.config['climatology_std_root']))):
            self._compute_climatology()
            
        # Apply climatology
        self._apply_climatology()
        
        # >>> Spatial slices
        self.data = space_time_slice(self.config, self.data, objects_to_slice = ('space'))
        self.labels = space_time_slice(self.config, self.labels, objects_to_slice = ('space'))
        self.context = space_time_slice(self.config, self.context, objects_to_slice = ('space'))
        self.indices = space_time_slice(self.config, self.indices, objects_to_slice = ('space'))
        
        # >>> Map of the dominant land use 
        self.map_dominant_context = np.zeros((len(self.config_context['features_selected']), self.context.dims['lat'], self.context.dims['lon']))
        for i, variable in enumerate(np.array(self.config_context['features'])[self.config_context['features_selected']]):
             self.map_dominant_context[i,:,:] = self.context[variable].values
        self.map_dominant_context = np.nanargmax(self.map_dominant_context, axis = 0)
        self.map_dominant_context = torch.from_numpy(self.map_dominant_context)
        #torch.save(self.map_dominant_context, os.path.join(self.config_context['root'], self.config_context['map_dominant_context_file']))
        #self.map_dominant_context = torch.load(os.path.join(self.config_context['root'], self.config_context['map_dominant_context_file']))
        
        ######################  General elements  ######################
        
        # >>> Define the sample reference
        # Take care, the samples are ordered according to drought percentages.
        self.samples_info = self._get_samples_info()

        # >>> Identification of drought events, location and em-dat id
        if self.period == 'test':
            self.mask_event_locations, self.events_ids = self._define_events()
    
    def _define_events(self):
        """Defines the evaluation events and provides: 
            1) the masks of events (of size: (n_masks, time, lat, lon)) and a
            2) vector with the ids that identify the events
        The masks have, as their values, the id of the event.
        Background is not identified as an object but appears in the masks with 0 values.
        The masks define for each event, its assigned location and time to evaluate.
        Prior to define the events, the clusters of pixels are treated as "objects".
        After checking the following conditions, the objects are named as "events". 
            1) remove_very_small = True, objects with less than min_pixels, are discarted
            2) connect_small_objects = True, objects separated less than min_dist are given the same label. 
                Min distance is spatio-temporal 
        Additionally: 
            *) same_for_all_times = True, the mask ignores when a drought starts or ends. 
                Static mask along the time dimension. This allows to visualize the evolution of the signal over a location.
            *) plot_events = True. Plots the events. 
            
        :return: mas of the event locations and vector of ids
        :rtype: torch.Tensor
        """
        print('-Defining the events')
        
        # Identify objects
        objects = label(self.labels.to_numpy().astype(int), background = 0, return_num = False, connectivity = 3)
        
        # Remove objects with less than a minimum of pixels
        if self.config['remove_very_small']: 
            print('--Removing very small objects. Min nº elements (px):', self.config['min_pixels'])
            
            for evID in np.unique(objects)[1:]:

                loc = (objects == evID)
                if np.sum(loc) <= self.config['min_pixels']:
                    objects = np.where(loc, 0, objects)
            
            # Rename the ids
            for newID, evID in enumerate(np.unique(objects)[1:]):
                objects = np.where(objects == evID, newID + 1, objects) # +1 to not use the zero (:=background)
        
        # Postprocess
        if self.config['connect_small_objects']:
            print('--Connecting small events. Distance (px):', self.config['min_dist'])
            
            # Get the centroids 
            obj_properties = regionprops(objects) # Labels with value 0 are ignored
            centroids = [np.array(obj_properties[i]['centroid']) for i in range(len(obj_properties))] 
            
            # Compute the Euler distance 
            distances = np.zeros((len(centroids), len(centroids)))
            for i, j in product(range(len(centroids)), range(len(centroids))):
                # counting spatio-temporal distance 
                # time has to be accounted to prevent two events getting the 
                # same label even if happening at different times
                distances[i, j] = np.linalg.norm(centroids[i] - centroids[j]) 
                
            # The matrix is simetric so we retain the corresponding 
            # upper triangular matrix / removing also the main diagonal
            # k = 0 is the main diagonal, k < 0 is below it and k > 0 is above
            distances[np.triu(distances, k = 1) == 0] = np.nan

            # Get the ids of the objects less than the distance
            # (+1 to avoid the zero as an index)
            id_objs1, id_objs2 = np.where(distances < self.config['min_dist'])
            id_objs1 += 1
            id_objs2 += 1
            
            # Unite the events and redefine ids
            pool_ids = []
            connected_objects = np.copy(objects)
            for evID in range(len(distances)):
                
                if evID not in pool_ids: # check for repetitions    
                    for near_evID in id_objs2[id_objs1 == evID]: # this checks for near objects
                            
                        connected_objects[objects == near_evID] = evID
                        pool_ids.append(near_evID)
                            
            # Change the object
            objects = np.copy(connected_objects)
                
        # Get the ids of the obects. Remove the zero (background)
        events_ids = np.unique(connected_objects)[1:].astype(dtype = np.uint8)
        
        # same_for_all_times
        if self.config['same_for_all_times']:
            print('--Setting the event region static. Same for all times' )
            
            # masks (variable = len(events_ids), time = 1, lat, lon)
            # Unite the events and redefine ids
            mask_event_locations = np.zeros((len(events_ids), 1, objects.shape[1], objects.shape[2]), dtype = np.uint8)
            for n_event, event_id in enumerate(events_ids):
                mask_event_locations[n_event] = (objects == event_id).any(axis = 0, keepdims = True) * event_id
                
        else: 
            # masks (variable = 1, time, lat, lon)
            mask_event_locations = objects[None, :].astype(dtype = np.uint8)
            
        if self.config['plot_events']:
            
            filename = os.path.join(self.config['root'], self.config['fdata'], 'Drought_n')
            tmp = mask_event_locations.sum(axis = (0, 1))
            for i in np.unique(tmp)[1:]:

                tmp_individual = np.where(tmp == i, 1, 0)
                for j in range(15):
                    tmp_individual = binary_dilation(tmp_individual)
                
                plt.figure()
                plt.imshow(tmp_individual)
                plt.title(str(i))
                plt.savefig(filename + str(i))
                plt.show()

        return torch.from_numpy(mask_event_locations), torch.from_numpy(events_ids)
    
    def _compute_climatology(self):
        """Compute climatology
        """
        print("--Computing statistics for standardization based on climatology")        
        
        os.mkdir(self.config['climatology_mean_root'])
        os.mkdir(self.config['climatology_std_root'])
        for feature in list(self.data_for_clim):

            # Mean and std
            monthly_feature = self.data_for_clim[feature].where(self.labels_for_clim == 0).groupby("time.month")
            climatology_mean_aux = monthly_feature.mean("time", skipna = True).to_dataset()
            climatology_std_aux = monthly_feature.std("time", skipna = True).to_dataset()

            # External save
            climatology_mean_aux.to_zarr(self.config['climatology_mean_root'] + '/' + feature + '.zarr')
            climatology_std_aux.to_zarr(self.config['climatology_std_root'] + '/' + feature + '.zarr')
        
    def _apply_climatology(self):
        """Apply climatology
        """
        print('--Removing trends from climatology')
        climatology_mean = xr.open_mfdataset(self.config['climatology_mean_root'] + '/*.zarr', engine = 'zarr')
        climatology_std = xr.open_mfdataset(self.config['climatology_std_root'] + '/*.zarr', engine = 'zarr')
        
        eps = 1e-7
        self.data = xr.apply_ufunc(
            lambda x, m, s: np.clip((x - m) / (3*s + eps), -1.0, 1.0),
            self.data.chunk({'time': 10}).groupby("time.month"),
            climatology_mean,
            climatology_std,
            dask = "parallelized",
        )
    
    def _get_samples_info(self):
        """Defines the indices of the samples and its characteristics (p_drought and distr)  

        :return: Samples information 
        :rtype: dict
        """
        print('-Defining sample reference object')
        
        filename = os.path.join(self.config['root'], 
                                self.config['fdata'], 
                                self.config['samples_info_file'] + '_' + self.period + '.pt') 
        if self.config['load_past'] and os.path.isfile(filename):
            
            print('--Loading past samples info') 
            samples_info = torch.load(filename)
            
        else:
            
            time_size_in = eval(self.config['input_size'])[2]
            time_size = self.data.dims['time'] - time_size_in
            
            if self.period == 'train' or not self.config['unfold_' + self.period]:
                
                # Get sizes
                lat_size_in, lon_size_in = eval(self.config['input_size'])[:2]
                n_samples = self.config['n_samples']
                
                # Define start indices
                start_time_idxs = torch.randint(0, time_size + 1, [n_samples, 1], dtype = torch.int)
                start_lat_idxs  = torch.randint(0, self.data.dims['lat'] - lat_size_in + 1, [n_samples, 1], dtype = torch.int)
                start_lon_idxs  = torch.randint(0, self.data.dims['lon'] - lon_size_in + 1, [n_samples, 1], dtype = torch.int)
                
            else:
                
                # Get sizes and overwrite number of samples
                lat_size_in, lon_size_in = (np.size(self.data.lat), np.size(self.data.lon))
                n_samples = time_size
                
                # Define start indices
                start_time_idxs = torch.arange(n_samples, dtype = torch.int)[:, None]
                start_lat_idxs = torch.zeros(n_samples, 1, dtype = torch.int)
                start_lon_idxs = torch.zeros(n_samples, 1, dtype = torch.int)
            
            # Define end indices
            end_time_idxs = start_time_idxs + time_size_in
            end_lat_idxs = start_lat_idxs + lat_size_in
            end_lon_idxs = start_lon_idxs + lon_size_in
            
            # Generate the idxs
            start_idxs = torch.cat((start_time_idxs, start_lat_idxs, start_lon_idxs), dim = 1)
            end_idxs = torch.cat((end_time_idxs, end_lat_idxs, end_lon_idxs), dim = 1)
            
            # Analize samples 
            p_drought_idxs = torch.empty(1)
            distr_idxs = torch.empty(1)
            if self.period == 'train' or not self.config['unfold_' + self.period]:
                
                # Compute the percentage of drought for each sample
                p_drought_idxs, distr_idxs = self.compute_pdrought_distr(start_idxs, end_idxs)
                
                # Non-maximum Suppression (NMS)
                b_nms = self.nms(p_drought_idxs, start_idxs, end_idxs) 
                
                # Slice the arrays according to the kept samples
                n_samples = len(b_nms) 
                start_idxs = start_idxs[b_nms, :]
                end_idxs = end_idxs[b_nms, :]
                p_drought_idxs = p_drought_idxs[b_nms, :]
                distr_idxs = distr_idxs[b_nms, :]
            
            # Save object 
            samples_info = {'n_samples': n_samples, 'start_idxs': start_idxs, 'end_idxs': end_idxs, 
                            'p_drought_idxs': p_drought_idxs, 'distr_idxs': distr_idxs}
            torch.save(samples_info, filename)
        
        return samples_info
    
    def compute_pdrought_distr(self, start_idxs, end_idxs):
        """Compute drought percentages and the input distribution

        :return: drought and distribution of the sample
        :rtype: torch.Tensor.float
        """
        print('--Computing the drought percentage in each sample and the distribution')
        
        # Identify the samples, loop once over all the data (not efficient)
        p_drought_idxs = []
        distr_idxs = []
        for index in range(self.config['n_samples']):
            
            ######################  Spatial and temporal positions of the sample  ########################
            start_time_idx, start_lat_idx, start_lon_idx = start_idxs[index]
            end_time_idx, end_lat_idx, end_lon_idx = end_idxs[index]
            
            # >>> Labels
            gt = torch.Tensor(self.labels.sel(lat = self.data.lat[start_lat_idx:end_lat_idx],
                                              lon = self.data.lon[start_lon_idx:end_lon_idx],
                                              time = self.data.time[start_time_idx:end_time_idx]).values > 0)[None, None, :]
            
            # We are interested in the percentage at the input
            # if interested at the output:
            # gt_output = crop_variable(self.config, gt, spatial_crop = True, temporal_crop = True).numpy()
            gt_ref = gt.numpy() 
            p_drought_idxs.append(gt_ref.sum() / gt_ref.size * 100) 
            
            # >>> Features
            features = self.data.sel(lat = self.data.lat[start_lat_idx:end_lat_idx],
                                     lon = self.data.lon[start_lon_idx:end_lon_idx],
                                     time = self.data.time[start_time_idx:end_time_idx]
                                     ).to_array().to_numpy()
            feature = np.mean(features, axis = 0).flatten()
            
            # Append the distribution 
            distr_idxs.append(feature)
        
        return (torch.FloatTensor(np.array(p_drought_idxs)[:, None]), 
                torch.FloatTensor(np.array(distr_idxs)))
            
    def nms(self, p_drought_idxs, start_idxs, end_idxs):
        """NMS

        :param p_drought_idxs: drought percentages
        :type p_drought_idxs: torch.Tensor.float
        :param start_idxs: list of start positions 
        :type start_idxs: list
        :param end_idxs: list of end positions 
        :type end_idxs: list
        :param time_correction: correction given the temporal effective receptive field 
        :type time_correction: int
        :return: indices to keep
        :rtype: list
        """
        print(f"--Performing non-maximum suppression with overlap = {self.config['overlap']}")
        
        p_drought_idxs = p_drought_idxs.flatten()
        order_idxs = torch.argsort(p_drought_idxs, descending = True)

        b_nms = []
        for idx_candidate in order_idxs:

            # Define the candidate box 
            # The (x1, y1, z1) position is at the top left corner,
            # the (x2, y2, z2) position is at the bottom right corner
            # take care that the order is this one, this are just indexes (not lat, lon or time)
            bb1 = {'x1': start_idxs[idx_candidate][2], 'x2': end_idxs[idx_candidate][2], 
                   'y1': start_idxs[idx_candidate][1], 'y2': end_idxs[idx_candidate][1], 
                   'z1': start_idxs[idx_candidate][0], 'z2': end_idxs[idx_candidate][0]} 
            
            discard = False
            judges_idxs = order_idxs[order_idxs != idx_candidate]
            for idx_judge in judges_idxs:
                
                # Define the judge box 
                bb2 = {'x1': start_idxs[idx_judge][2], 'x2': end_idxs[idx_judge][2], 
                       'y1': start_idxs[idx_judge][1], 'y2': end_idxs[idx_judge][1], 
                       'z1': start_idxs[idx_judge][0], 'z2': end_idxs[idx_judge][0]} 
                
                # Compute iou between boxes
                iou = get_3Diou(bb1, bb2)
                #print('idx_candidate:', idx_candidate, 'vs', idx_judge, 'iou', iou)
                
                # Check overlapping condition and % of drought
                if iou > self.config['overlap']:
            
                    # We apply the criterium of retaining only the sample with more drought content.    
                    # Discard the "candidate" if its drought content is lower than the one from the "judge"
                    # Discards two non drought samples if they do not have drought content
                    if p_drought_idxs[idx_judge] >= p_drought_idxs[idx_candidate]:
                        discard = True
                        break
                        
            if not discard:
                b_nms.append(idx_candidate)
                
        return b_nms
    
    def __getitem__(self, index):
        """Get item from database
        
        :return: set of samples for the batch
        :rtype: dict
        """
        #print('Sample number:', index)
            
        ######################  Spatial and temporal positions of the sample  ########################
                
        start_time_idx, start_lat_idx, start_lon_idx = self.samples_info['start_idxs'][index]
        end_time_idx, end_lat_idx, end_lon_idx = self.samples_info['end_idxs'][index]

        size_lat = end_lat_idx - start_lat_idx
        size_lon = end_lon_idx - start_lon_idx

        #################################  Predefine variables  ##################################

        loc_size = (eval(self.config['input_size'])[2], size_lat, size_lon) #time, lat, lon
        features = torch.zeros(((len(self.config['features_selected']),) + loc_size)) 
        masks = torch.zeros(((len(self.config['features_selected']),) + loc_size)) 
        gt = torch.zeros(((1,) + loc_size)) 
        dindices = torch.zeros(((len(self.config_indices['features_selected']),) + loc_size)) 
        
        #################################  Fill Variables  ##################################
        
        # >>> Features and Masks
        for i in np.arange(len(self.config['features_selected'])):
            
            # Features (variables, time, lat, lon)
            feature = self.data[np.array(self.config['features'])[self.config['features_selected'][i]]] \
                [start_time_idx:end_time_idx, start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx]
            feature = torch.Tensor(feature.transpose('time','lat','lon').values) # we transpose to make sure (but it comes with the correct order)
            features[i, :np.shape(feature)[0], :, :] = feature
            
            # Masks (variables, time, lat, lon)
            mask = torch.logical_not(torch.isnan(feature))
            masks[i, :np.shape(mask)[0], :, :] = mask  
        
        # Fill missing values with 0s
        features[torch.isnan(features)] = 0 
        x = features
        
        # Get a common mask for all variables (1 = data, 0 = no_data) 
        masks = (torch.prod(masks, 0, keepdim = True) == 1)
        
        # >>> Labels
        item_date = self.data['time'][start_time_idx:end_time_idx].values
        gt_found = np.where(np.in1d(item_date,self.labels['time']))[0] # Test whether each element of a 1-D array is also present in a second array.
        gt[:, gt_found,:,:] = torch.Tensor(self.labels.sel(lat = self.data.lat[start_lat_idx:end_lat_idx],
                                                           lon = self.data.lon[start_lon_idx:end_lon_idx],
                                                           time = item_date[gt_found]).transpose('time','lat','lon').values > 0) 
        # we transpose to make sure (but it comes with the correct order). 
        # Also, the conditional >0 converts the previosly GDIS drought values into a boolean mask 
        
        # >>> Indices
        for i in np.arange(len(self.config_indices['features_selected'])):
            
            # Features (variables, time, lat, lon)
            dindex = self.indices[np.array(self.config_indices['features'])[self.config_indices['features_selected'][i]]] \
                [start_time_idx:end_time_idx, start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx]
            dindex = torch.Tensor(dindex.values) 
            dindices[i, :np.shape(dindex)[0], :, :] = dindex
        
        #################################  Misc. variables/procedures  ##################################
        
        # >>> Data augmentation
        if self.config['augment'] and self.period == 'train':
            
            x = self._augment(x)
            masks = (self._augment(masks * 1.0).round() == 1)
            gt = self._augment(gt * 1.0).round()
        
        #################################  Out  ##################################
        
        if self.period == 'test':
            
            # Expand the dimension of context (add temporal and variable dimensions, = 1 (it does not increase more than one sample))
            map_dominant_context = self.map_dominant_context.expand(gt.size())
            
            # If the masks have more than one timestep (not collapsed), 
            # slice them to the same size as the other input variables
            # if not, expand the temporal dimension to the same size by 
            # repeating in that dimension 
            if self.mask_event_locations.size(1) > 1:
                mask_event_locations = self.mask_event_locations[:, start_time_idx:end_time_idx]
            
            else:
                mask_event_locations = self.mask_event_locations.repeat((1, eval(self.config['input_size'])[2], 1, 1))
                
            return x, masks, gt, map_dominant_context, self.events_ids, mask_event_locations
    
        else:
            return x, masks, gt, index, dindices
    
    def _augment(self, var):
        """Augmentation
        """
        transform_vflip = T.RandomVerticalFlip(p = self.config['augment_prob'])
        var = transform_vflip(var)
        
        return var
    
    def __len__(self):
        """Returns the length of the dataset (number of frames <- temporal instances)
        """
        return self.samples_info['n_samples']