#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D BBox IOU
"""
import numpy as np

def get_3Diou(bb1, bb2, z_area=True):
    """
    Calculate the Intersection over Union (IoU) of two 3D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'z1', 'z2'}
        The (x1, y1, z1) position is at the top left corner,
        the (x2, y2, z2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'z1', 'z2'}
        The (x1, y1, z1) position is at the top left, z1 corner,
        the (x2, y2, z2) position is at the bottom right, z2 corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert (bb1['x1'] < bb1['x2']).all()
    assert (bb1['y1'] < bb1['y2']).all()
    assert (bb1['z1'] < bb1['z2']).all()
    assert (bb2['x1'] < bb2['x2']).all()
    assert (bb2['y1'] < bb2['y2']).all()
    assert (bb2['z1'] < bb2['z2']).all()
    
    # determine the coordinates of the intersection cuboid
    x_left = np.maximum(bb1['x1'], bb2['x1'])
    y_top = np.maximum(bb1['y1'], bb2['y1'])
    z_0 = np.maximum(bb1['z1'], bb2['z1'])
    x_right = np.minimum(bb1['x2'], bb2['x2'])
    y_bottom = np.minimum(bb1['y2'], bb2['y2'])
    z_t = np.minimum(bb1['z2'], bb2['z2'])

    zero_intersection = np.logical_or(np.logical_or(x_right < x_left, y_bottom < y_top), z_t < z_0)

    # The intersection of three axis-aligned 3D bounding boxes is always an
    # axis-aligned 3D bounding box
    if z_area: 
        intersection_area = (x_right - x_left) * (y_bottom - y_top) * (z_t - z_0) 
        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1']) * (bb1['z2'] - bb1['z1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1']) * (bb2['z2'] - bb2['z1'])
    else: 
        intersection_area = (x_right - x_left) * (y_bottom - y_top) 
        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
        
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    """
    # normalizing to account to the max IoU than can be achieved for the fixed box size (fized image resolution)
    min_bb_areas = np.minimum(bb1_area, bb2_area)
    max_bb_areas = np.maximum(bb1_area, bb2_area)
    # if 
    eps = 1e-7
    iou = iou * 1/((min_bb_areas/max_bb_areas) + eps)
    """
    # if zero intersection, set iou as zero
    iou[zero_intersection] = 0.0
    assert (iou >= 0.0).any()
    assert (iou <= 1.0).any()
    return iou
