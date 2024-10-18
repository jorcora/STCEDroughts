#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY
import numpy as np

# SCIPY
import scipy

# PYTORCH
import torch
import torch.nn.functional as F

# SCIKIT IMAGE
from skimage.segmentation import slic, checkerboard_level_set
from skimage.morphology import binary_dilation, binary_erosion
from skimage.measure import label

# SKLEARN
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.preprocessing import KernelCenterer

# MATPLOTLIB 
import matplotlib as mpl
import matplotlib.pyplot as plt

# SEABORN
import seaborn as sns

# MONAI
from monai.metrics import compute_average_surface_distance

class SPXsLABEL: 
    def __init__(self, model_config):
        """Initialializes the SPX class 

        :param model_config: model configuration
        :type model_config: dict
        """
        self.split_method = model_config['SPX']['split_method']
        self.npixels = model_config['SPX']['npixels']
        self.metric = model_config['SPX']['metric']
        self.softLabels = model_config['SPX']['softLabels']
        self.normalize_output = model_config['SPX']['normalize_output']
        self.noise_form = model_config['SPX']['robustness']['noise_form'] 
        self.noise_level = model_config['SPX']['robustness']['noise_level']
        self.activate_every = model_config['SPX']['robustness']['activate_every'] 
        self.deactivate_every = model_config['SPX']['robustness']['deactivate_every']
        self.id_process = model_config['SPX']['robustness']['id_process']
        self.n_process = model_config['SPX']['robustness']['n_process']
        self.idx_labels_corrupted = []
        self.labels_corrupted = []
        
    def transform_labels(self, output, labels): 
        """Get the auxiliary labels to transform the output

        :param output: output of the model
        :type output: np.ndarray
        :param labels: labels
        :type labels: np.ndarray
        :return: auxiliary labels 
        :rtype: np.ndarray
        """        
        
        # 1) Criterion of segmentation. 
        # Divide the output into supersegments. 
        # Respect the boundary of the droughts region provided by the labels
        self.spxs_idxs = self.get_spx_ids(output, labels, split_method = self.split_method, npixels = self.npixels)
        
        # 2) Get the value of the spx according to the selected metric
        spxs_mtrs = self.get_spx_metr(output, metric = self.metric)
            
        # 3) Criterion of transformation. Transform the regions
        if self.softLabels:
            
            new_labels = labels.copy()
            new_labels = np.where(labels == 0, spxs_mtrs[0], spxs_mtrs[1]) 
        
        else: 
            new_labels = self.modify_regions(self.spxs_idxs, spxs_mtrs, labels)
            
        return new_labels
    
    def get_spx_ids(self, var, labels, split_method = 'slic', npixels = 25):
        """supersegments var according to method

        :param var: variable to supersegment
        :type var: np.ndarray
        :param labels: labels to supersegment independently each class
        :type labels: np.ndarray
        :return: supersegmented var
        :rtype: np.ndarray
        """
        if split_method == 'slic':
            
            self.split_mparams = {'compactness': 10,  # this is the starting value for slico
                                 'spacing': [8, 1, 1], 'sigma': 0.0, 'max_num_iter': 1000, # num inter 10
                                 'enforce_connectivity': True, 'start_label': 1, 'slic_zero': True,
                                 'convert2lab': False, 
                                 'channel_axis': 0}
            
            # Inside the drought region
            mask = labels
            n_segments = int(np.max([2, int(np.ceil(np.sum(mask) / self.npixels))])) # encouraging superpixels of that many pixels 
            spxs_in = slic(var[None, :], n_segments = n_segments, mask = mask, **self.split_mparams)
            
            # Outside the drought region
            try: # to prevent when all the regions have drought labels
                mask = np.logical_not(labels)
                n_segments = int(np.max([2, int(np.ceil(np.sum(mask) / self.npixels))])) # encouraging superpixels of that many pixels 
                spxs_out = slic(var[None, :], n_segments = n_segments, mask = mask, **self.split_mparams) 
                spxs_out = spxs_out + (int(np.max(spxs_in)) * mask)  
            except:
                spxs_out = np.empty_like(spxs_in)
            
            # Combined
            spxs_idxs = np.where(labels, spxs_in, spxs_out)
            
        if split_method == 'fix_tiles':
            
            square_size = int(np.sqrt(npixels))
            pattern = checkerboard_level_set(np.shape(var), square_size = square_size) 
            
            # Inside the drought region
            try: # to prevent when all the regions have non drought labels
                mask = labels
                pattern_in = np.where(mask, pattern, -1) 
                spxs_in = label(pattern_in.astype(int), background = -1, return_num = False, connectivity = 1)
            except:
                spxs_in = np.empty_like(labels)
                
            # Outside the drought region
            try: # to prevent when all the regions have drought labels
                mask = np.logical_not(labels)
                pattern_out = np.where(mask, pattern, -1) 
                spxs_out = label(pattern_out.astype(int), background = -1, return_num = False, connectivity = 1)
                spxs_out = spxs_out + (int(np.max(spxs_in)) * mask)  
            except:
                spxs_out = np.empty_like(spxs_in)

            # Combined
            spxs_idxs = np.where(labels, spxs_in, spxs_out)
        
        return spxs_idxs
    
    def get_spx_metr(self, output, other_var = np.zeros(0), metric = 'mean'):
        """Compute a metric over each superpixel

        :param output: predictions of the model
        :type output: np.ndarray
        :param other_var: auiliary variable, defaults to np.zeros(0)
        :type other_var: np.ndarray, optional
        :param metric: name of the metric to compute, defaults to 'mean'
        :type metric: str, optional
        :return: new array with the values given by the metric in each superpixel/location
        :rtype: np.ndarray
        """
        if metric == 'mean': 
            spxs_mtrs = np.zeros(((2,) + np.shape(self.spxs_idxs)))
            for idx_spx in np.unique(self.spxs_idxs):
                
                loc = (self.spxs_idxs == idx_spx) 
                spxs_mtrs[:, loc] = np.mean(output[loc]) 

        if metric == 'negative_mean': 
            spxs_mtrs = np.zeros(((2,) + np.shape(self.spxs_idxs)))
            for idx_spx in np.unique(self.spxs_idxs):
                
                loc = (self.spxs_idxs == idx_spx) 
                spxs_mtrs[:, loc] = -np.mean(output[loc]) 
                
        if metric == 'uniform_distribution':
            
            spxs_mtrs = np.zeros(((2,) + np.shape(self.spxs_idxs)))
            spxs_mtrs[:] = np.ones_like(output) * 1/2
            
        if metric == 'negative_preds':
            
            spxs_mtrs = np.zeros(((2,) + np.shape(self.spxs_idxs)))
            spxs_mtrs[:] = -1 * output
            
        if metric == 'positive_preds':
            
            spxs_mtrs = np.zeros(((2,) + np.shape(self.spxs_idxs)))
            spxs_mtrs[:] = output

        if metric == 'conf_score_entr':

            # the relation between the entropy (Shannon) of each spx and
            # the entropy of an uniform distribution, vect(u). 
            # Where u_i = 1/C, C := number of classes = 2
            Hu_i = -1/2 * np.log(1/2)
            spxs_mtrs = np.zeros((np.shape(self.spxs_idxs)))
            for idx_spx in np.unique(self.spxs_idxs):
            
                loc = (self.spxs_idxs == idx_spx)
                pk = output[loc]
                H_spx = -np.sum(pk * np.log(pk)) # The maximum likelihood estimator (also called the “plug-in” or “naive” estimator)
                #loc_entr = loc_entr + (np.sum(loc) - 1) / (2 * np.sum(loc)) # The Miller-Madow corrected estimator 
                spxs_mtrs[loc] = 1 - H_spx / (Hu_i * len(pk)) 

        if metric == 'conf_score_entr_pointwise':

            Hu_i = -1/2 * np.log(1/2)
            H_spx = -output * np.log(output)
            spxs_mtrs = 1 - H_spx / Hu_i
                
        return spxs_mtrs
    
    def define_epsilon(self, output, other_var = np.zeros(0), time_counter = 0,
                       beta = 0.5, time_weight = True, metric = 'conf_score_entr'):
        """defines the epsilon mixing factor 

        :param output: outputs of the model
        :type output: np.ndarray
        :param other_var: auxiliary variable, defaults to np.zeros(0)
        :type other_var: _type_, optional
        :param time_counter: global trust score, defaults to 0
        :type time_counter: int, optional
        :param beta: decaying factor of the global trust score, defaults to 0.5
        :type beta: float, optional
        :param time_weight: activates or not the global trust score, defaults to True
        :type time_weight: bool, optional
        :param metric: metric to compute the auxiliary term, defaults to 'conf_score_entr'
        :type metric: str, optional
        :return: epsilon mixing factor
        :rtype: np.ndarray
        """
        # Confidence score: 
        conf_score = self.get_spx_metr(output, other_var = other_var, metric = metric)
        
        # Temporal score:
        temp_score = 1
        if time_weight:
            temp_score = 1 / (1 + np.exp(-beta * time_counter)) 
        
        # Define the eps 
        eps = conf_score * temp_score
          
        return eps
    
    def modify_regions(self, spxs_idxs, spxs_mtrs, labels, 
                       max_iterations = 1000, convergence = True, 
                       activate_every = 1, deactivate_every = 1,
                       n_process = 1, id_process = 0):
        
        new_labels = labels.copy()
        idx_list = np.ones((max_iterations + 1, 2)) * np.nan
        iterations = np.arange(max_iterations)
        activate_deactivate = (np.array([1/activate_every * iterations, 
                                         1/deactivate_every * iterations]).T % 1 == 0)
                                         
        for i in range(max_iterations):
            print(i)
            # Get positions and metrics
            outer_idxs, outer_mtr, _, _ = self.get_boundary_regions(new_labels, spxs_idxs, spxs_mtrs)
            border_idxs = outer_idxs
            border_mtrs = outer_mtr
            value = 1
            # Select according to proportion
            # Also prevent an error when a group (outer or inner) is empty. (loc is empty and this gives an error)
            # in this case we allow the approach to activate/deactivate till the complementary option can be executed
            #if activate_deactivate[i, j] and len(border_idxs) > 0:
            if len(border_idxs) > 0:
                    
                    # Get the idx of the metric and transform the location
                    border_idx = np.random.choice(border_idxs, size = n_process, p = border_mtrs/np.sum(border_mtrs))  #border_idxs[np.argmax(border_mtrs)] 
                    border_idx = border_idx[id_process]
                    loc = (spxs_idxs == border_idx)
                    new_labels[loc] = value 

            # Get positions and metrics
            _, _, inner_idxs, inner_mtr = self.get_boundary_regions(new_labels, spxs_idxs, spxs_mtrs)
            border_idxs = inner_idxs
            border_mtrs = inner_mtr
            value = 0
            # Select according to proportion
            # Also prevent an error when a group (outer or inner) is empty. (loc is empty and this gives an error)
            # in this case we allow the approach to activate/deactivate till the complementary option can be executed
            #if activate_deactivate[i, j] and len(border_idxs) > 0:
            if len(border_idxs) > 0:
                    
                    # Get the idx of the metric and transform the location
                    border_idx = np.random.choice(border_idxs, size = n_process, p = border_mtrs/np.sum(border_mtrs))  #border_idxs[np.argmax(border_mtrs)] 
                    border_idx = border_idx[id_process]
                    loc = (spxs_idxs == border_idx)
                    new_labels[loc] = value 
                
        return new_labels * 1.0
    
    def get_boundary_regions(self, new_labels, spxs_idxs, spxs_mtrs):
        
        # Get the metrics corresponding to the actual class
        spxs_mtrs = np.where(new_labels, spxs_mtrs[1], spxs_mtrs[0])
        
        # Get the adjacent elements to the drought boundary
        # (elements inside and outside the drought region) 
        dlabels = binary_dilation(new_labels) 
        elabels = binary_erosion(new_labels) 
        region = np.logical_and(dlabels, np.logical_not(elabels))
        region_idxs = spxs_idxs[region]
        region_mtrs = spxs_mtrs[region]
        
        # Get the unique elements positions 
        _, uorder = np.unique(region_idxs, return_index = True) 
        region_idxs = region_idxs[np.sort(uorder)]
        region_mtrs = region_mtrs[np.sort(uorder)]
        
        # Separate the regions into outer and inner regions 
        elements_outer = []
        elements_inner = []
        for idx  in region_idxs:
            
            elements = new_labels[(spxs_idxs == idx)]
            elements_outer.append(np.sum(elements == 0)) # outer region
            elements_inner.append(np.sum(elements == 1)) # inner region
        
        elements_outer = np.array(elements_outer)
        elements_inner = np.array(elements_inner)
        
        # Get the regions 
        outer_idxs = region_idxs[elements_outer >= elements_inner]
        inner_idxs = region_idxs[elements_outer < elements_inner]
        
        # And the corresponding metrics values
        outer_mtrs = region_mtrs[elements_outer >= elements_inner]
        inner_mtrs = region_mtrs[elements_outer < elements_inner]
        
        return outer_idxs, outer_mtrs, inner_idxs, inner_mtrs
    
    def plot_3D(self, data, path2save, cmap_name = 'Reds',
                lims = np.array([0, 0.25, 0.5, 0.75, 1.0]),
                anglex = 30, angley = 35,
                disp_grid = False, disp_axis = 'on',
                print_format = 'png'):
        
        # Reorder the axis and flip for correct display 
        data = np.flip(np.moveaxis(data, 1, 2), axis = 2)
        
        # Define the colors
        N = len(lims) - 1
        assert N > 1, 'Num boundaries should be more than 2s'
        color_mat = np.empty(data.shape, dtype = object)
        cmap = plt.cm.get_cmap(cmap_name, N) 
        for i in range(N):
            
            color_name = mpl.colors.rgb2hex(cmap(i))
            color_mat[data >= lims[i]] = color_name
        
        # Plot the figure
        fig = plt.figure(figsize = (12, 12))
        ax = fig.add_subplot(1,1,1, projection = '3d')
        ax.voxels(data, facecolors = color_mat, edgecolors = color_mat, alpha=1)
        
        ax.set(xlim = (0, data.shape[0]), ylim = (0, data.shape[1]), zlim = (0, data.shape[2]))
        ax.set(xlabel = 'time', ylabel = 'lon', zlabel = 'lat') 
        ax.invert_xaxis()
        ax.view_init(anglex, angley)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(disp_grid, linestyle='--', alpha=0.7)
        plt.axis(disp_axis)
        
        plt.tight_layout()
        plt.savefig(path2save + f'.{print_format}', 
                    format = print_format, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        plt.close(fig) 
        
    def plot_transform(self, output, labels, new_labels, eps, combined_labels, path2save, print_format = 'png'):

        # Plot superpixel segmentation
        names = ['_PRED', '_SPX', '_EPS', '_1-EPS', '_ORIGINAL', '_COMBINED']
        DATA = [output[0], new_labels[0], eps[0], 1-eps[0], labels[0], combined_labels[0]]
        cmap_objs = plt.cm.get_cmap('Reds', 100)
        cmap_hyps = plt.cm.get_cmap('plasma', 100) #plasma inferno, PuOr
        cmaps = [cmap_objs, cmap_objs, cmap_hyps, cmap_hyps, cmap_objs, cmap_objs]  
        for i in range(len(names)):
            
            fig, ax = plt.subplots(1, 1, figsize = (12, 12))
            im = ax.imshow(DATA[i], cmap = cmaps[i])
            im.set_clim(0,1)
            cbar = fig.colorbar(im, ax = ax, location = 'bottom', pad = 0.05)
            cbar.ax.locator_params(nbins = 4)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.savefig(path2save + f'{names[i]}.{print_format}', 
                        format = print_format, bbox_inches = 'tight', pad_inches = 0)
            plt.show()
            plt.close(fig)
    
    def voxel_metrics(self,labels, new_labels, metric = 'distance'):
        """
        noise estimation between the labels
        or distance estimation 
        """
        if metric == 'noise': 
            voxel_metric = 0
        
        if metric == 'distance': 
            
            # Prepare data in the required monai package format
            # move depth (time) dimension to the end and one_hot transform the variables
            new_labels = torch.moveaxis(new_labels, 2, -1)
            labels = torch.moveaxis(labels, 2, -1)
            one_hot_new_labels = F.one_hot(new_labels.type(torch.LongTensor), num_classes = 2) 
            one_hot_labels = F.one_hot(labels.type(torch.LongTensor), num_classes = 2)
            voxel_metric = compute_average_surface_distance(one_hot_new_labels, one_hot_labels)
            
        return voxel_metric
