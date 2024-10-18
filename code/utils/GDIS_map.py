#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# NUMPY 
import numpy as np

# CARTOPY
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
                                
# MATPLOTLIB
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# SKIMAGE
from skimage.measure import regionprops
from skimage.draw import rectangle
from skimage.morphology import cube, dilation, erosion


def define_nonoverlaping_3Dforms(map_events, labels, border3Dforms = (1,1,1), margin = 5, plot_3Dforms = True, save_path = ''):
    """Creates bounding boxes around the events

    :param map_events: array defining the events location
    :type map_events: np.ndarray
    :param labels: labels 
    :type labels: np.ndarray
    :param border3Dforms: size of the bounding box (time, lat, lon), defaults to (1,1,1)
    :type border3Dforms: tuple, optional
    :param margin: min margin to add between bounding boxes, defaults to 5
    :type margin: int, optional
    :param plot_3Dforms: to plot the bounding boxes reference, defaults to True
    :type plot_3Dforms: bool, optional
    :param save_path: path to save the reference images, defaults to ''
    :type save_path: str, optional
    :return: bounding boxes array
    :rtype: np.ndarray
    """
    # Auxiliar indices bounding boxes correction
    # To avoid overlapping problems. Indices relating to directions: [up, down, left, right]
    axis_idxs = [0, 0, 1, 1] # defines if we are moving in latitude or longitude
    maxmin_idxs = [-1, 0, -1, 0] # if we are interested in maximum or minimum value
    margins = margin * np.array([1, -1, 1, -1]) # for visualization purposes
        
    # Transform the events in boxes
    boxed_map_events = np.zeros_like(map_events)
    for num_event, event_id in enumerate(np.unique(map_events)[1:]):
        # Locate the event and get its properties
        loc_event = (map_events == event_id).any(axis = 1)
        coordinates = (labels * loc_event).nonzero()
        t_start, t_end = min(coordinates[0]), max(coordinates[0])
        lat_start, lat_end = min(coordinates[1]), max(coordinates[1])
        lon_start, lon_end = min(coordinates[2]), max(coordinates[2])

        # then we define the bbox
        boxed_map_events[t_start:t_end+1, num_event, lat_start:lat_end+1, lon_start:lon_end+1] = event_id

    # Define a bounding box around each event
    forms3D_regions = np.zeros_like(labels)
    for event_id in np.unique(map_events)[1:]:
        
        # Locate the event and get its properties
        loc_event = (map_events == event_id).any(axis = 1)
        
        # an event can have multiple episodes
        # we define the bounding box by connecting all episodes
        # first we get the coordinates and store them for later use
        # Notice than when doing the slice, the end point is not included
        # We add a plus one to correct that in the slices. 
        coordinates = (labels * loc_event).nonzero()
        t_start, t_end = min(coordinates[0]), max(coordinates[0])
        lat_start, lat_end = min(coordinates[1]), max(coordinates[1])
        lon_start, lon_end = min(coordinates[2]), max(coordinates[2])

        # then we define the bbox
        bbox = np.zeros_like(labels)
        bbox[t_start:t_end+1, lat_start:lat_end+1, lon_start:lon_end+1] = 1
        
        # Check for conflicts (overlapping spatial region with other bounding boxes)
        # Locate the other events and define the projection limits of the current event in all directions
        loc_others = np.logical_and(boxed_map_events != event_id, boxed_map_events != 0).any(axis = (0, 1)) 
        dflt_lims = [0, -1, 0, -1] # default lims. Sets the scope of the expansion to the full map 
        plim_values = [lat_start, lat_end, lon_start, lon_end]
        plims = [[0, lat_start+1, lon_start, lon_end+1], # up
                    [lat_end, -1, lon_start, lon_end+1], # down
                    [lat_start, lat_end+1, 0, lon_start+1], # left 
                    [lat_start, lat_end+1, lon_end, -1]]  # right
        
        for nn, lim in enumerate(plims): 
    
            dummy = np.zeros_like(loc_others)
            dummy[lim[0]:lim[1], lim[2]:lim[3]] = 1
            conflicting_loc = np.logical_and(loc_others, dummy) 

            if conflicting_loc.any():
    
                mm_coord = np.sort(conflicting_loc.nonzero()[axis_idxs[nn]])[maxmin_idxs[nn]]
                mm_coord = round(abs((plim_values[nn] - mm_coord)/2)) + np.min([plim_values[nn], mm_coord]) # middle point
                mm_coord += margins[nn] # for visualization purposes we add a buffer
                mm_coord += abs(dflt_lims[nn]) # this amendment is for the slice. The end positions get a plus one
                dflt_lims[nn] = mm_coord 
        
        # Define the mask
        mask = np.zeros_like(labels)
        mask[:, dflt_lims[0]:dflt_lims[1], dflt_lims[2]:dflt_lims[3]] = 1
                
        # Define the dilated
        dilated_bbox = dilation(bbox, footprint = np.ones(border3Dforms))
        corrected_bbox = np.logical_and(dilated_bbox, mask)
        forms3D_regions[corrected_bbox] = event_id

    # Plot the new perimeter
    if plot_3Dforms:
        plt.figure()
        plt.imshow(labels.any(axis = 0))
        plt.contour(forms3D_regions.sum(axis = 0))
        plt.savefig(save_path + '/3Dforms_space')
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(forms3D_regions.any(axis = (1,2)))
        plt.savefig(save_path + '/3Dforms_time')
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(labels.any(axis = (1,2)))
        plt.savefig(save_path + '/3Dforms_time_original')
        plt.show()
        plt.close()

    return (forms3D_regions != 0)

def geoplot2d(ax, im_base, im_extent, 
              im_contour = np.zeros(0),
              bbox_contour = np.zeros(0),  
              colors_ref = 'Reds', set_colorbar = True):
    """This function adds georeferenced axis and plots an image "im_base" with spatial 
    coordinates defined by im_extend. If provided, it also plots the contourns 
    of a two auxiliary images which is assumed to have one level of values.  

    :param ax: axes object
    :type ax: matplotlib.pyplot.axes
    :param im_base: image to be plotted over a geocoordinated grid
    :type im_base: np.ndarray
    :param im_extent: tuple defining the lat lon coordinates
    :type im_extent: tuple
    :param im_contour: contour image, defaults to np.zeros(0)
    :type im_contour: np.ndarray, optional
    :param bbox_contour: contour image, defaults to np.zeros(0)
    :type bbox_contour: np.ndarray, optional
    :param colors_ref: _description_, defaults to 'Reds'
    :type colors_ref: str, optional
    :param set_colorbar: _description_, defaults to True
    :type set_colorbar: bool, optional
    :return: _description_
    :rtype: matplotlib.pyplot.figure
    """                   
    # Define color palette
    new_reds = plt.cm.get_cmap(colors_ref, 100) 
    
    # use gray color to mark 'nan' values
    new_reds.set_bad(color = 'lightsteelblue') 
    
    # Plot base image
    # data comes from a regular latitude/longitude grid == as a PlateCarree projection
    # The projection maps meridians to vertical straight lines of constant spacing 
    # (for meridional intervals of constant spacing), and circles of latitude to 
    # horizontal straight lines of constant spacing (for constant intervals of parallels)
    
    # If we don't tell the projecttion the data is defined, matplotlib/cartopy 
    # is going to assume the same cs as the one used to define the plot
    # we specify the coordinate system of the data by the argument "transform". data_crs = ccrs.PlateCarree()
    ax.set_extent(im_extent, crs = ccrs.PlateCarree()) 
    im = ax.imshow(np.flipud(im_base), extent=im_extent, origin='lower', 
                   cmap=new_reds, transform = ccrs.PlateCarree()) 
    
    if set_colorbar:
        axins = inset_axes(ax, # parent axes
            width = "100%",  # width = 5% of parent_bbox width
            height = "2.5%",  # height : 50% % of parent_bbox width
            loc = 'center', #loc specifies where inside the box the legend sits.
            bbox_to_anchor = (0, -0.525, 1, 1), #(x0, y0, width, height)
            bbox_transform = ax.transAxes,
            borderpad = 0,
            )
        im.set_clim(0,1)
        plt.colorbar(mappable = im, cax = axins, orientation='horizontal') # axins.cbar_axes[0].colorbar(p)
    
    # Add contours
    if im_contour.any():
        ax.contour(np.flipud(im_contour), extent = im_extent, linestyles = '-', 
                       colors = ['k'], alpha = 1, linewidths = 2, 
                       origin = 'lower', transform = ccrs.PlateCarree())
    if bbox_contour.any():
        ax.contour(np.flipud(bbox_contour), extent = im_extent, linestyles = '-', 
                       colors = ['k'], alpha = 1, linewidths = 2,  
                       origin = 'lower', transform = ccrs.PlateCarree())

    # Add axis features
    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = False, 
                  linewidth = 2, color='gray', alpha = 0.5, linestyle = ':') 
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator([0, 45])
    gl.ylocator = mticker.FixedLocator([0, 45])
    gl.ylocator = LatitudeLocator(nbins = 4)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.ylabel_style = {'size': 15, 'color': 'black', 'weight': 'bold'}
    gl.xlabel_style = {'size': 15, 'color': 'black', 'weight': 'bold'}
    gl.xpadding = -2
    gl.ypadding = -2
    ax.coastlines(resolution='50m')

    return im
    
def plot_GDIS_map(title, y_hat, masks, labels, dataset, dataset_config, save_path,
                  bbox_contour = np.zeros(0), drought_id = None, 
                  set_colorbar = True, print_format = 'png'):
    """Plot GDIS result map

    :param title: tile of the image
    :type title: str
    :param y_hat: values to plot
    :type y_hat: np.ndarray
    :param masks: values to mask
    :type masks: np.ndarray
    :param labels: object of reference
    :type labels: np.ndarray
    :param dataset: GDIS dataset
    :type dataset: torch.utils.data.Dataset
    :param dataset_config: configuration of the dataset
    :type dataset_config: dict
    :param save_path: path to save the image
    :type save_path: str
    :param bbox_contour: boxes around positions, defaults to np.zeros(0)
    :type bbox_contour: np.ndarray, optional
    :param drought_id: id for getting one event at a time, defaults to None
    :type drought_id: int, optional
    :param set_colorbar: activates or not the colorbar, defaults to True
    :type set_colorbar: bool, optional
    :param print_format: sets the format for printing, defaults to 'png'
    :type print_format: str, optional
    """
    # Turn places with no data into nan
    y_hat[masks==0] = np.nan
    
    # Adjust the resolution according to missing positions 
    # due to not using padding in the model
    size_in = list(eval(dataset_config['GDIS']['input_size']))
    size_out = list(eval(dataset_config['GDIS']['output_size']))
    resolution_adjust = int((size_in[0] - size_out[0])/2) * dataset_config['GDIS']['resolution']    
    resolution_adjustvismargin = dataset_config['GDIS']['vismargin'] * dataset_config['GDIS']['resolution']   

    # Define image extension
    if drought_id: 
        # Get id_disasterno, lat1, lat2, lon1, lon2 of the drought
        _, lat1, lat2, lon1, lon2  = dataset.droughts_table.iloc[drought_id].values
        
        # Border 
        border = 5
        lat1 += border
        lat2 -= border
        lon1 -= border
        lon2 += border
        
        # img_extent = [minlon, maxlon, minlat, maxlat] coordinates where the data is defined
        im_extent = (lon1 + resolution_adjust, 
                     lon2- resolution_adjustvismargin,
                     lat2 + resolution_adjustvismargin,
                     lat1 - resolution_adjust) 
        
    else:
        # img_extent = [minlon, maxlon, minlat, maxlat] coordinates where the data is defined
        im_extent = (eval(dataset_config['GDIS']['lon_slice'])[0] + resolution_adjust, 
                     eval(dataset_config['GDIS']['lon_slice'])[1] - resolution_adjustvismargin,
                     eval(dataset_config['GDIS']['lat_slice'])[1] + resolution_adjustvismargin,
                     eval(dataset_config['GDIS']['lat_slice'])[0] - resolution_adjust) 
    
    # Central latitude and longitude
    central_longitude = (im_extent[1] - abs(im_extent[0]))/2
    central_latitude = (im_extent[3] - abs(im_extent[2]))/2
    
    # Process the variables if selecting a particular drought
    if drought_id:
        # Define mask
        im_lat = np.logical_and(dataset.data.lat.values <= lat1,
                                dataset.data.lat.values >= lat2)
        im_lon = np.logical_and(dataset.data.lon.values >= lon1,
                                dataset.data.lon.values <= lon2)
        
        # Turn into a grid
        exten_mask_lon, extent_mask_lat = np.meshgrid(im_lon, im_lat)
        
        # Combine
        extent_mask = np.logical_and(exten_mask_lon, extent_mask_lat)
        
        # Get the combined lat and lon
        extent_mask_lat, extent_mask_lon = np.where(extent_mask)
        
        # Adapt variables        
        y_hat = y_hat[extent_mask, extent_mask_lon].reshape(len(np.unique(extent_mask)), len(np.unique(extent_mask_lon)))
        masks = masks[extent_mask, extent_mask_lon].reshape(len(np.unique(extent_mask)), len(np.unique(extent_mask_lon)))
        labels = labels[extent_mask, extent_mask_lon].reshape(len(np.unique(extent_mask)), len(np.unique(extent_mask_lon)))
    
    # Plot    
    fig = plt.figure(figsize = (10, 14)) 
    ax = plt.subplot(1, 1, 1, projection = ccrs.LambertConformal(central_longitude = central_longitude, central_latitude = central_latitude)) 
    #plt.title(title, fontsize = 20)
    _ = geoplot2d(ax,
                  y_hat,
                  im_extent,
                  im_contour = labels,
                  bbox_contour = bbox_contour, 
                  set_colorbar = set_colorbar)
    plt.savefig(save_path + '/' + title + f'.{print_format}', 
                format = print_format, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    plt.close(fig = fig)
