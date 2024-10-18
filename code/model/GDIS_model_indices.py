#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jordi
"""

# TIME
import time

# NUMPY 
import numpy as np

# PYTORCH
import torch

# PYTORCH LIGHTNING
import pytorch_lightning as pl

# SKLEARN 
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, average_precision_score

# UTILS
from utils import compute_loss, crop_variable, SPXsLABEL, measures_test, M2P_P2M

# ARCHITECTURES
from .arch import *

class MY_MODEL(pl.LightningModule):
  def __init__(self, experiment_config, dataset_config, model_config):
    super().__init__()
        
    # Save hyperparameters
    self.save_hyperparameters()
      
    # Configs
    self.experiment_config = experiment_config
    self.dataset_config = dataset_config
    self.model_config = model_config
      
    # Load the model
    self.model = globals()[model_config['arch']['name']](model_config)
      
    # Define the mechanism
    self.spxTransform = SPXsLABEL(model_config)
    self.eps = model_config['SPX']['eps']
    self.counter = 0

    # Load the loss
    self.loss = globals()['compute_loss']
      
    # Register of the training auroc for computing the
    # relative metrices between train and validation
    self.train_auroc = 0
    
    # Register of the samples with added noise for robustness tests
    self.num_samples_available = self.dataset_config['len_loader_train']
    print('self.num_samples_available', self.num_samples_available)
    
    self.num_samples_to_train = int(np.round(self.model_config['SPX']['robustness']['noise_level'] * self.num_samples_available))
    print('self.num_samples_to_train', self.num_samples_to_train)

    dummy_ = np.zeros(self.num_samples_available)
    dummy_[:self.num_samples_to_train] = 1

    permutations = []
    for nexp in range(model_config['SPX']['robustness']['n_process']):
      
      rng = np.random.default_rng(nexp + 10)
      permutation = rng.permutation(dummy_)
      permutations.append(permutation)

    self.label_to_train = permutations[self.model_config['SPX']['robustness']['id_process']]  
    print('self.label_to_train', self.label_to_train)
    
    self.generator = torch.Generator()
    self.generator.manual_seed(self.model_config['SPX']['robustness']['gen_id'])


  def forward(self, x, step_samples):
    return self.model(x, step_samples)
  
  def shared_step(self, x, masks, labels, mode, index = 0, dindices = 0, step_samples = 1):

    # Evaluate
    if self.dataset_config['GDIS'][f'unfold_{mode}']:
        res = self.evaluate_map(x, masks, labels, mode)
    
    else: 
        res = self.evaluate_patch(x, masks, labels, mode, index = index, 
                                  dindices = dindices, step_samples = step_samples)
    
    # Loggers
    self.log(f'{mode}_loss', res['loss'], on_step = True, 
             on_epoch = True, prog_bar = False, logger = True)

    return res 
    
  def evaluate_map(self, x, masks, labels, mode):
      
    # Size references
    input_size = eval(self.dataset_config['GDIS']['input_size'])
    output_size = eval(self.dataset_config['GDIS']['output_size']) 
    
    # Create classes to fold/unfold input maps
    m2p_p2m_x = M2P_P2M(x, input_size, output_size, stride = output_size[0], generate_map = True)
    m2p_p2m_masks = M2P_P2M(masks, input_size, output_size, stride = output_size[0], generate_map = False)
    m2p_p2m_labels = M2P_P2M(labels, input_size, output_size, stride = output_size[0], generate_map = False)

    # Perform evaluate for each patch
    loss_patches = []
    #mean_time = []
    for idx in m2p_p2m_x.idxs:
        #start_time = time.time()
        res = self.evaluate_patch(m2p_p2m_x.get_patch(idx), 
                                  m2p_p2m_masks.get_patch(idx),
                                  m2p_p2m_labels.get_patch(idx), 
                                  mode)
        #inference_time_secs_it = round((time.time() - start_time), 2)
        #print(idx, inference_time_secs_it)
        #mean_time.append(inference_time_secs_it)

        # Agregate results
        loss_patches.append(res['loss'])
        m2p_p2m_x.reconstruct_map(res['output'], idx)
        #m2p_p2m_labels.reconstruct_map(res['labels'], idx)
    
    #print('mean_time', np.mean(mean_time))  
    #print('num_patches', len(m2p_p2m_x.idxs))           
    
    # Save results
    res = {}
    res['loss'] =  torch.mean(torch.stack(loss_patches))
    res['output'] = m2p_p2m_x.get_reconstructed_map()
    #res['labels'] = m2p_p2m_labels.get_reconstructed_map()
    res['labels'] = labels[:, :, self.dataset_config['GDIS']['idt_ref_in'], 
                           int((input_size[0] - output_size[0])/2):, 
                           int((input_size[0] - output_size[0])/2):] 
    
    return res

  def evaluate_patch(self, x, masks, labels, mode, index = 0, dindices = 0, step_samples = 1): #step_samples = 100
    
    # Move variables to device
    x = x.to(labels.device) 
    masks = masks.to(labels.device) 
    labels = labels.to(labels.device)
    
    # Forward
    logits = self.forward(x, step_samples).to(labels.device)
    
    # Convert output to scores
    labels_hat = torch.sigmoid(logits)

    # Adapt variables to the new output size                                       
    labels_hat = crop_variable(self.dataset_config['GDIS'], labels_hat, spatial_crop = True, temporal_crop = True) 
    labels = crop_variable(self.dataset_config['GDIS'], labels, spatial_crop = True, temporal_crop = True)
    masks = crop_variable(self.dataset_config['GDIS'], masks, spatial_crop = True, temporal_crop = True) 
    masks = torch.where(logits == -999, 0, masks)
    
    # Apply the SPX mechanism
    if mode == 'train' and self.model_config['SPX']['SPX'] and labels.any(): 

      # Obtain the new labels
      spx_labels = self.spxTransform.transform_labels(labels_hat[0, 0].detach().cpu().numpy(), 
                                                      labels[0, 0].detach().cpu().numpy())
      spx_labels = torch.from_numpy(spx_labels).unsqueeze(0).unsqueeze(0).to(labels.device)
      
      # Compute the hyperparameter, epsilon
      x_cropped = crop_variable(self.dataset_config['GDIS'], x, spatial_crop = True, temporal_crop = True)
      hyp = self.spxTransform.define_epsilon(labels_hat[0, 0].detach().cpu().numpy(), 
                                              other_var = x_cropped[0,:].cpu().numpy(),
                                              time_counter = self.counter, beta = self.eps,
                                              time_weight = self.model_config['SPX']['time_weight'],
                                              metric = self.model_config['SPX']['metric_eps_score'])
      hyp = torch.from_numpy(hyp).unsqueeze(0).unsqueeze(0).to(labels.device)
      
      # Compute the loss
      loss = self.loss(self.model_config, labels_hat, labels, masks, spx_labels = spx_labels, eps = hyp)  
      self.counter += 1 # Counter for the SPX method
      
      # Plots
      if index in [10, 12, 15] and self.current_epoch in [0,1,2,3,4,5,6,7,8,9,10]: 
          # Define the path of the image      
          path2save = (self.experiment_config['images_spx_path'] + '/sample_' +
                        str(int(index)) + '_epoch_'+ str(self.current_epoch))
          combined_labels = ((1 - hyp) * labels + hyp * spx_labels).detach()
          
          # output, labels, new_labels
          self.spxTransform.plot_transform(labels_hat[0, 0].detach().cpu().numpy(), 
                                           labels[0, 0].detach().cpu().numpy(), 
                                           spx_labels[0, 0].cpu().numpy(), 
                                           hyp[0, 0].detach().cpu().numpy(),
                                           combined_labels[0, 0].detach().cpu().numpy(),
                                           path2save, print_format = self.experiment_config['Arguments']['print_format'])
          
    else: 
      loss = self.loss(self.model_config, labels_hat, labels, masks)      

    return {'loss': loss, 'output': labels_hat.detach(), 'labels': labels, 'masks': masks}   
  
  def training_step(self, batch, batch_idx):
    x, masks, labels, index, _  = batch 
    
     
    if self.label_to_train[batch_idx] and self.current_epoch > 1: 
      
      dummy = torch.randint_like(labels.detach().clone(), low=0, high=2)
      idx = torch.randperm(dummy.nelement(), generator = self.generator)
      dummy = dummy.view(-1)[idx].view(dummy.size())  
      transf_labels = torch.where(labels == 0, dummy, labels)    
      res = self.shared_step(x, masks, transf_labels, mode = 'train', index = index, step_samples = 100) 
      return res
      
    else:
        
      res = self.shared_step(x, masks, labels, mode = 'train', index = index, step_samples = 100) 
      return res 

  def validation_step(self, batch, batch_idx):
    x, masks, labels, _, _ = batch 
    res = self.shared_step(x, masks, labels, mode = 'val', step_samples = 100)
    return res
    
  def test_step(self, batch, batch_idx):
    x, masks, labels, map_dominant_context, mask_event_ids, mask_event_locations = batch 
    res = self.shared_step(x, masks, labels, mode = 'test', step_samples = 1)
    res_test = measures_test(res, self.dataset_config, masks, map_dominant_context, mask_event_ids, mask_event_locations)
    return res_test
  
  def training_epoch_end(self, training_step_outputs):
    self.metrics_and_log(training_step_outputs, mode = 'train')
      
  def validation_epoch_end(self, validation_step_outputs):
    self.metrics_and_log(validation_step_outputs, mode = 'val')
      
  def test_epoch_end(self, test_step_outputs):
    self.metrics_and_log(test_step_outputs, mode = 'test')
  
  def metrics_and_log(self, mode_step_outputs, mode):
        
    # Average Loss
    avg_loss = np.mean(np.hstack([batch['loss'].detach().cpu().numpy() for batch in mode_step_outputs]))    
        
    if mode == 'test':
        
        print('')
        print('\nResults. Accumulated output and labels')
        outputs_masked = np.ndarray.flatten(np.concatenate([batch['outputs_masked'].cpu().numpy() for batch in mode_step_outputs]))
        labels_masked = np.ndarray.flatten(np.concatenate([batch['labels_masked'].cpu().numpy() for batch in mode_step_outputs]))

        # AUROC     
        agg_AUROC = roc_auc_score(labels_masked, outputs_masked)
        self.log(f'avg_{mode}_aggAUROC', agg_AUROC, on_epoch=True, prog_bar=False, logger=True)
        print(f'\nAggregated res AUROC = {agg_AUROC}')

        # AVERAGE PRECISION 
        agg_AP = average_precision_score(labels_masked, outputs_masked)
        self.log(f'avg_{mode}_aggAP', agg_AP, on_epoch=True, prog_bar=False, logger=True)
        print(f'\n--Aggregated res AP = {agg_AP}')

        # F1 (given an optimal threshold)
        fpr, tpr, thresholds = roc_curve(labels_masked, outputs_masked)
        threshold = thresholds[np.argmax(tpr - fpr)]
        agg_f1_cutoff = f1_score(labels_masked, (outputs_masked >= threshold)*1.0, zero_division = 'warn')
        self.log(f'avg_{mode}_F1_cutoff', agg_f1_cutoff, on_epoch = True, prog_bar=False, logger=True)
        print(f'\n--Aggregated res F1_cutoff = {agg_f1_cutoff}')
        
        # PSNR (Higher is better)
        mse = np.mean((labels_masked - outputs_masked)**2)
        PSNR = 10 * np.log10((1 ** 2) / mse) # data_range = 1. 
        self.log(f'avg_{mode}_PSNR', PSNR, on_epoch = True, prog_bar=False, logger=True)
        print(f'\n--Peak Signal to Noise Ratio (PSNR) = {PSNR}')

        # CORRELATION
        R = np.corrcoef(labels_masked, outputs_masked)[0, 1]
        self.log(f'avg_{mode}_R', PSNR, on_epoch = True, prog_bar=False, logger=True)
        print(f'\n--Correlation (R) = {R}')

        ######################################################     
        print('\n CUBE METRICES')
        # spatial AUROC all 
        avg_auroc = np.nanmean([batch['auroc_step'] for batch in mode_step_outputs]) 
        print(f'AUROC_cube_spatial_all = {avg_auroc}')

        # spatial AUROC all - variance    
        var_auroc = np.nanvar([batch['auroc_step'] for batch in mode_step_outputs])
        print(f'AUROC_cube_spatial_all_variance = {var_auroc}')

        # spatial AUROC all - standard deviation    
        std_auroc = np.nanstd([batch['auroc_step'] for batch in mode_step_outputs])
        print(f'AUROC_cube_spatial_all_std = {std_auroc}')

        # AP (Average Precision score) 
        avg_average_precision = np.nanmean(np.hstack([batch['aucpr_step'] for batch in mode_step_outputs]))
        print(f'AP_cube_spatial_all = {avg_average_precision}')

        # spatial PA variance
        var_average_precision = np.nanvar(np.hstack([batch['aucpr_step'] for batch in mode_step_outputs]))
        print(f'AP_cube_spatial_all_variance = {var_average_precision}')
        
        # spatial PA standard deviation    
        std_average_precision = np.nanstd(np.hstack([batch['aucpr_step'] for batch in mode_step_outputs]))
        print(f'AP_cube_spatial_all_std = {std_average_precision}')
        
        # F1
        avg_f1 = np.nanmean(np.hstack([batch['f1_step'] for batch in mode_step_outputs]))
             
        print('\n INSTANCE METRICES')
        # spatial AUROC all
        avg_instance_test_auroc = np.nanmean(np.hstack([batch['instance_test_auroc'] for batch in mode_step_outputs]))
        print(f'AUROC_instance_spatial_all = {avg_instance_test_auroc}')
             
        # TEMPORAL AUROC (only instances)
        # get the events ids
        mask_event_ids = mode_step_outputs[0]['mask_event_ids'] #get them from one batch as they are the same for all
        # temporal AUROC all
        for event in mask_event_ids:
            signals_outputs_temp = []
            signals_labels_temp = []
            temporal_auroc = np.nan
            for batch in mode_step_outputs: 
                signals_outputs_temp.append(batch['test_signal_outputs'][event])
                signals_labels_temp.append(batch['test_signal_labels'][event])
            signals_labels_temp = np.array(signals_labels_temp)
            signals_outputs_temp = np.array(signals_outputs_temp)
            if not signals_labels_temp.all() and signals_labels_temp.any(): # avoid event that is all true in the 5 fold
                temporal_auroc = roc_auc_score(signals_labels_temp, signals_outputs_temp)
            print('AUROC_instance_temporal_'+str(event)+'= ', temporal_auroc)
        
        print('\n CM MEASURES')
        # Performance measures
        avg_TP = np.nanmean(np.hstack([batch['TP'] for batch in mode_step_outputs]))
        avg_TN = np.nanmean(np.hstack([batch['TN'] for batch in mode_step_outputs]))
        avg_FP = np.nanmean(np.hstack([batch['FP'] for batch in mode_step_outputs]))
        avg_FN = np.nanmean(np.hstack([batch['FN'] for batch in mode_step_outputs]))
        print('TP_cube_spatial_all= ', avg_TP)
        print('TN_cube_spatial_all= ', avg_TN)
        print('FP_cube_spatial_all= ', avg_FP)
        print('FN_cube_spatial_all= ', avg_FN)

    else:
        # Outputs for all batches
        outputs = np.ndarray.flatten(np.concatenate([batch['output'].cpu().numpy() for batch in mode_step_outputs]))
        labels = np.ndarray.flatten(np.concatenate([batch['labels'].cpu().numpy() for batch in mode_step_outputs]))

        # Masks for positions where we have variable values 
        masks = (np.ndarray.flatten(np.concatenate([batch['masks'].cpu().numpy() for batch in mode_step_outputs])) == 1)
        
        # AUROC
        avg_auroc = roc_auc_score(labels[masks], outputs[masks])
        
        # AUC PR (Average Precision score) 
        avg_average_precision = average_precision_score(labels[masks], outputs[masks])
        
        # F1
        avg_f1 = f1_score(labels[masks], outputs[masks].round(), zero_division = 'warn')
        
        # F1 given an optimal threshold
        fpr, tpr, thresholds = roc_curve(labels[masks], outputs[masks]) # thesholds in decreasing order
        threshold = thresholds[np.argmax(tpr - fpr)]
        avg_f1_cutoff = f1_score(labels[masks], (outputs[masks] >= threshold)*1.0, zero_division = 'warn')
        self.log(f'avg_{mode}_f1_cutoff', avg_f1_cutoff, on_epoch = True, prog_bar=True, logger=True)  
             
        # DIFF
        if mode == 'train':
            self.train_auroc = avg_auroc
        
        if mode == 'val':
            diff_auroc = self.train_auroc - avg_auroc
            self.log('diff_avg_auroc', diff_auroc, on_epoch = True, prog_bar=True, logger=True)
            
            ratio_avg_auroc = avg_auroc/self.train_auroc
            self.log('ratio_avg_auroc', ratio_avg_auroc, on_epoch = True, prog_bar=True, logger=True)
            
            ratio_auroc = avg_auroc/self.train_auroc 
            mselection = 1/avg_loss * ratio_auroc  # so higher values are better  
            self.log('mselection', mselection, on_epoch = True, prog_bar=True, logger=True)
                
    # Loggers
    self.log(f'avg_{mode}_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
    self.log(f'avg_{mode}_auroc', avg_auroc, on_epoch = True, prog_bar=True, logger=True)
    self.log(f'avg_{mode}_f1', avg_f1, on_epoch = True, prog_bar=True, logger=True)
    self.log(f'avg_{mode}_average_precision', avg_average_precision, on_epoch = True, prog_bar=True, logger=True)  
  
  def configure_optimizers(self):
    """Configure optimizers
    """
    trainable_params = filter(lambda p: p.requires_grad, self.parameters())
    optimizer = torch.optim.Adam(trainable_params, 
                                 lr = self.model_config['optimizer']['lr'], 
                                 weight_decay = self.model_config['optimizer']['wd'],
                                 amsgrad = False)

    return {'optimizer': optimizer}
