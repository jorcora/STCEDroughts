#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi cortés-andrés
"""

# NUMPY
import numpy as np

# SKLEARN 
from sklearn.metrics import confusion_matrix, roc_curve

# PANDAS 
import pandas as pd

# MATPLOTLIB
import matplotlib.pyplot as plt

# SEABORN
import seaborn as sn

def get_threshold(preds, labels, num = 50, metric = 'Sensitivity (TPR)'):
    """Optimizes the CM metric defined by finding the optimal threshold 
    Available metrics: 
            'TP', 'FP', 'TN', 'FN',
            'Sensitivity (TPR)', 'Specificity (TNR)', 
            'Positive Predictive Value (PPV)', 'Negative Predictive Value (NPV)',
            'Overall Acc',
            'Threat score (TS)'
    """
    minv = np.min(preds)
    maxv = np.max(preds)
    thres_opt = np.linspace(minv, maxv, num = num, endpoint = True)
    metric_values = []
    for threshold in thres_opt:
        
        tmp_preds = (preds >= threshold) * 1.0
        res = cm_measures(labels, tmp_preds)
        metric_values.append(res[metric])

    thres = thres_opt[np.argmax(metric_values)]
    
    print('Metric to optimize:', metric)
    print('Optimal threshold for binarization:', thres)
    
    return thres

def plot_GDIS_cm(preds, labels, masks, save_path, thres = 0.5, 
                 find_threshold = True, metric = 'Sensitivity (TPR)', 
                 print_format = 'png'):
    
    # Flatten the predictions
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()
    masks = np.array(masks).flatten()
    
    # Avoid nan positions 
    preds = preds[masks]
    labels = labels[masks] 
    
    # Find the optimal threshold and binarize the output scores
    if find_threshold:
        thres = get_threshold(preds, labels, metric = metric)
    preds = (preds >= thres) * 1.0
    
    # Define the CM matrix
    classes = ('No drought', 'Drought')
    res = cm_measures(labels, preds)  
    cmat = np.array([[res['TN'], res['FP']], 
                     [res['FN'], res['TP']]]) 
    cmat = cmat / cmat.sum(axis = 1, keepdims = True)
    df_cmat = pd.DataFrame(cmat, index = [c for c in classes], columns = [c for c in classes])
    
    # Plot
    fig = plt.figure(figsize=(12, 12))
    sn.heatmap(df_cmat, annot = True)
    plt.ylabel('GT')
    plt.xlabel('Predicted')
    plt.title('Threshold:' + str(round(thres, 2)))
    plt.savefig(save_path + '/CM_Optimize_' + metric + '_' + str(find_threshold) + f'.{print_format}', 
                format = print_format, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    plt.close(fig = fig)
    
def cm_measures(labels, outputs):
    """
    Function to compute different measures from the confusion matrix
    Some definitions: https://www.nature.com/articles/s41598-022-09954-8
    """
    TN, FP, FN, TP  = confusion_matrix(labels, outputs).ravel()    
    
    # Sensitivity (also called hit rate, recall, or true positive rate)
    # denotes the rate of positive samples correctly classified. 
    # Bounded [0, 1]. 1 = all "true class" predicted. 0 = no "true class" predicted. 
    TPR = TP/(TP + FN)
    
    # Specificity or true negative rate
    # denotes the rate of negative samples correctly classified. 
    # Bounded [0, 1]. 1 = all "false class" predicted. 0 = no "false class" predicted.
    TNR = TN/(TN + FP) 
    
    # Precision (can refer to PPV or NPV): 
    # Proportion of the retrieved samples which are relevant. 
    # Proportion of true samples predicted for a class vs all samples predicted as belonging to that class.
    # Bounded [0, 1]. 1 = all samples in the class predicted. 0 = no correct predition in the corresponding class.
    # "Among the samples assigned to class 'A' by the model, how many are correctly assigned?" 
    
    # Positive predictive value
    if TP + FP != 0:
        PPV = TP/(TP + FP)
    else:
        PPV = np.nan
    # Negative predictive value
    if TN + FN != 0:
        NPV = TN/(TN + FN)
    else: 
        NPV = np.nan
        
    # Overall accuracy
    # Ratio between the correctly classified samples and the total number of samples in the evaluation dataset.
    # Bounded [0, 1]. 1 = all samples positive and negative correctly classified. 0 = no correct classification.
    ACC = (TP + TN)/(TP + FP + FN + TN)
    
    # Threat score (TS) (also called Critical Success Index (CSI))
    # ratio between the number of correctly predicted positive samples against the sum of correctly predicted positive samples and all incorrect predictions.
    # It takes into account both false alarms and missed events in a balanced way, and excludes only the correctly predicted negative samples. 
    # Bounded [0, 1]. 1 = false predictions in either class. 0 = correctly classified positive samples.
    TS = TP / (TP + FN + FP)
    
    # Youden's Index (J) is a single statistic that captures the performance of a dichotomous diagnostic test.
    # Defines an optimal threshold. Bounded [0, 1].
    J = TPR + TNR - 1
    
    return  {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'Sensitivity (TPR)': TPR, 'Specificity (TNR)': TNR, 
            'Positive Predictive Value (PPV)': PPV, 'Negative Predictive Value (NPV)': NPV,
            'Overall Acc': ACC, 'Threat score (TS)': TS, "Youden's Index": J}
