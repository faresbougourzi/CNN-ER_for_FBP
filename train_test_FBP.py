#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:15:21 2021

@author: bougourzi
"""

# Face Beauty Prediction Project
# Bougourzi Fares

from FBP_Dataloader import Beauty_Db, Beauty_Db2im
from models import REXINCET
from utils import dy_huber_loss, dy_smooth_l1_loss, dy_tukey_loss
from utils import MAE_distance, PC_mine

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio
import os

from tqdm import tqdm

from scipy.stats import pearsonr

import argparse


def Train_FBP():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='REXINCET',
                        help='training model= REXINCET, Inception, ResneXt')
    parser.add_argument('--Fold', type=int, default=6,
                        help='1-5 stand for the 5 folds and 6 for 60-40 data splitting') 
    parser.add_argument('--LossFnc', type=str, default= 'Dy_Huber',
                        help='loss Function can be: MSE, Dy_Huber, Dy_ParamSmoothL1, Dy_Tukey') 
    parser.add_argument('--modelsavepath', type=str, default= "./Models/",
                        help='The path where to save the trained model')
    parser.add_argument('--device', type=str, default= "cuda:0",
                        help='The GPU device if available') 
    parser.add_argument('--Nepochs', type=int, default=30,
                        help='The Number of Epochs')
    parser.add_argument('--batchsize', type=int, default=15,
                        help='The Batch Size')   
    parser.add_argument('--initlr', type=float, default=0.0001,
                        help='The Batch Size')     
    
    opt = parser.parse_args()
     
    
    if opt.model == 'Inception' or opt.model == 'ResneXt':
        if opt.model == 'Inception':
            img_size = 299
            model = torchvision.models.inception_v3(pretrained=True)
            model.fc = nn.Linear(2048, 1)
        else:
            img_size = 224 
            model = torchvision.models.resnext50_32x4d(pretrained=True)
            model.fc = nn.Linear(2048, 1)
        # Transforms
        train_transform = transforms.Compose([
                transforms.ToPILImage(mode='RGB'),
                transforms.Resize((img_size,img_size)),
                transforms.RandomRotation(degrees = (-5,5)),
                transforms.ToTensor()
        ])    
        test_transform = transforms.Compose([
                transforms.ToPILImage(mode='RGB'),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
        ]) 
        
        # Data splits loading
        train_set = Beauty_Db(
                root='./'
                ,train = 'train_fold'+ str(opt.Fold)+ '.pt'
                ,transform = train_transform
        )
                
        test_set = Beauty_Db(
                root='./'
                ,train = 'test_fold'+ str(opt.Fold)+ '.pt'
                ,transform = test_transform
        ) 
    else:  # REXINCET model
        # create model instance
        model = REXINCET()
        # Transforms
        train_transform = transforms.Compose([
                transforms.ToPILImage(mode='RGB'),
                transforms.RandomRotation(degrees = (-5,5)),        
                transforms.ToTensor()
                ])    
                
        test_transform = transforms.Compose([
                transforms.ToPILImage(mode='RGB'),
                transforms.ToTensor()
                ])
         # Data splits loading
        train_set = Beauty_Db2im(
                root='./'
                ,train = 'train_fold'+ str(opt.Fold)+ '.pt'
                ,transform = train_transform
        )
                
        test_set = Beauty_Db2im(
                root='./'
                ,train = 'test_fold'+ str(opt.Fold)+ '.pt'
                ,transform = test_transform
        ) 
        
    # Loss Function
    if opt.LossFnc == 'MSE':
       criterion = nn.MSELoss()
     
    elif opt.LossFnc == 'Dy_Huber':
      criterion =  dy_huber_loss
      sigma_max = 0.7
      sigma_min = 0.3
    elif opt.LossFnc == 'Dy_ParamSmoothL1':
      criterion = dy_smooth_l1_loss  
      sigma_max = 0.7
      sigma_min = 0.3
      
    else:
      criterion = dy_tukey_loss
      sigma_max = 2
      sigma_min = 1
      
    models_save_path = opt.modelsavepath
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path) 
        
                
    device = torch.device(opt.device) 
        
    model = model.to(device)     
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = opt.batchsize, shuffle = True)      
    validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)
    
    
    train_MAE = []
    train_RMSE = []
    train_PC = []
    train_epsilon = []
    
    test_MAE = []    
    test_RMSE = []
    test_PC = []
    test_epsilon = []
    epoch_count = []
    
    pc_best = -2
    
    
    
    """
    Training anf Testing Loop
    There are three networks options REXINCET, Inception, ResneXt
    Four Loss Functions: MSE, Dynamic Huber, Dynamic ParamSmoothL1, Dynamic Tukey
    Each Scenario has slightly different code
    """ 
    for epoch in range(opt.Nepochs):
        lr = opt.initlr
        if epoch>19:
            lr = opt.initlr * 0.1  
        if epoch>29:
            lr = opt.initlr * 0.01       
        
        
        epoch_count.append(epoch)
        optimizer = optim.Adam(model.parameters(), lr =lr)
        total_loss = 0
        total_loss_val = 0
    
        labels2_tr = []
        labels_pred_tr = []
        epsi_tr = [] 
        
        
        if opt.LossFnc == 'MSE':
            
            if opt.model == 'REXINCET':
               
                for batch in tqdm(train_loader):        
                    images1, images2, labels, epsil = batch
                    images1 = images1.float().to(device)
                    images2 = images2.float().to(device)
                    labels = labels.to(device)
                    torch.set_grad_enabled(True)
                    model.train()
                    preds = model(images1, images2)
            
                    loss = criterion(preds.squeeze(1), labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    
                    total_loss += loss.item()            
                    labels2_tr.extend(labels.cpu().numpy())
                    labels_pred_tr.extend(preds.detach().cpu().numpy()) 
                    epsi_tr.extend(epsil.numpy())                    
                    del images1; del images2; del labels              
            
            
            elif opt.model == 'ResneXt':
        
                for batch in tqdm(train_loader):        
                    images, labels, epsil = batch
                    images = images.float().to(device)
                    labels = labels.to(device)
                    torch.set_grad_enabled(True)
                    model.train()
                    preds = model(images)
                    
                    loss = criterion(preds.squeeze(1), labels)            
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()            
    
                    total_loss += loss.item()            
                    labels2_tr.extend(labels.cpu().numpy())
                    labels_pred_tr.extend(preds.detach().cpu().numpy()) 
                    epsi_tr.extend(epsil.numpy())                    
                    del images; del labels                     
                    
            else:
                for batch in tqdm(train_loader):        
                    images, labels, epsil = batch
                    images = images.float().to(device)
                    labels = labels.to(device)
                    torch.set_grad_enabled(True)
                    model.train()
                    preds, preds2 = model(images)
                    
                    loss = criterion(preds.squeeze(1), labels)            
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    
                    total_loss += loss.item()            
                    labels2_tr.extend(labels.cpu().numpy())
                    labels_pred_tr.extend(preds.detach().cpu().numpy()) 
                    epsi_tr.extend(epsil.numpy())                    
                    del images; del labels                
                
                
        else: 
            sigma = sigma_min + (1/2)* (sigma_max - sigma_min ) * (1+ np.cos (np.pi * ((epoch+1)/ opt.Nepochs)))
            
            if opt.model == 'REXINCET':
        
                for batch in tqdm(train_loader):        
                    images1, images2, labels, epsil = batch
                    images1 = images1.float().to(device)
                    images2 = images2.float().to(device)
                    labels = labels.to(device)
                    torch.set_grad_enabled(True)
                    model.train()
                    preds = model(images1, images2)
            
                    loss = criterion(preds.squeeze(1), labels, sigma)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    
                    total_loss += loss.item()            
                    labels2_tr.extend(labels.cpu().numpy())
                    labels_pred_tr.extend(preds.detach().detach().cpu().numpy()) 
                    epsi_tr.extend(epsil.numpy())                    
                    del images1; del images2; del labels                                          
                    
                    
            elif  opt.model == 'ResneXt':
        
                for batch in tqdm(train_loader):        
                    images, labels, epsil = batch
                    images = images.float().to(device)
                    labels = labels.to(device)
                    torch.set_grad_enabled(True)
                    model.train()
                    preds = model(images)
            
                    loss = criterion(preds.squeeze(1), labels, sigma)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    
                    total_loss += loss.item()            
                    labels2_tr.extend(labels.cpu().numpy())
                    labels_pred_tr.extend(preds.detach().cpu().numpy()) 
                    epsi_tr.extend(epsil.numpy())                    
                    del images; del labels 
                    
            else:
                for batch in tqdm(train_loader):        
                    images, labels, epsil = batch
                    images = images.float().to(device)
                    labels = labels.to(device)
                    torch.set_grad_enabled(True)
                    model.train()
                    preds, preds2 = model(images) 
                    
                    loss = criterion(preds.squeeze(1), labels, sigma)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    
                    total_loss += loss.item()            
                    labels2_tr.extend(labels.cpu().numpy())
                    labels_pred_tr.extend(preds.detach().cpu().numpy()) 
                    epsi_tr.extend(epsil.numpy())                    
                    del images; del labels 
                    
                    
        labels2_ts = []
        labels_pred_ts = []
        epsi_ts =  []
        
        if opt.model == 'REXINCET' :
            for batch in tqdm(validate_loader): 
                images1, images2, labels, epsil = batch
                images1 = images1.float().to(device)
                images2 = images2.float().to(device)
                labels = labels.to(device)
                model.eval()
                with torch.no_grad():
                    preds = model(images1, images2)
        
                # loss = criterion(preds.squeeze(1), labels)                
               
                total_loss_val += MAE_distance(preds.squeeze(1), labels)
                labels2_ts.extend(labels.cpu().numpy())
                labels_pred_ts.extend(preds.detach().cpu().numpy())
                epsi_ts.extend(epsil.numpy())                     
                del images1; del images2; del labels            
            
        else:
            for batch in tqdm(validate_loader): 
                images, labels, epsil = batch
                images = images.float().to(device)
                labels = labels.to(device)
                model.eval()
                with torch.no_grad():
                    preds = model(images)
        
                # loss = criterion(preds.squeeze(1), labels)                
               
                total_loss_val += MAE_distance(preds.squeeze(1), labels)
                labels2_ts.extend(labels.cpu().numpy())
                labels_pred_ts.extend(preds.detach().cpu().numpy())
                epsi_ts.extend(epsil.numpy())                     
                del images; del labels
            
        labels_pred_ts = np.squeeze(np.array(labels_pred_ts))
        labels2_ts = np.array(labels2_ts)
        labels_pred_tr = np.squeeze(np.array(labels_pred_tr))
        labels2_tr = np.array(labels2_tr)        
        
        epsi_ts = np.array(epsi_ts)
        epsi_tr = np.array(epsi_tr)
        
        
        test_MAE.append(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
        test_RMSE.append(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
        test_PC.append(float(PC_mine(labels_pred_ts, labels2_ts)))
        test_epsilon.append(float(np.mean(1-np.exp(-((labels_pred_ts - labels2_ts)**2)/(2*(epsi_ts)**2)))))
        
        train_MAE.append(float(np.mean(np.abs(labels_pred_tr - labels2_tr))))
        train_RMSE.append(float(np.sqrt(np.mean((labels_pred_tr - labels2_tr)**2))))
        train_PC.append(float(PC_mine(labels_pred_tr, labels2_tr)))
        train_epsilon.append(float(np.mean(1-np.exp(-((labels_pred_tr - labels2_tr)**2)/(2*(epsi_tr)**2)))))
        print('Ep: ', epoch, 'PC_tr: ', PC_mine(labels_pred_tr, labels2_tr), 'PC_ts: ',  PC_mine(labels_pred_ts, labels2_ts),'MAE_tr: ', total_loss/len(train_set), 'MAE_ts: ', total_loss_val/len(test_set), 'loss_tr:', total_loss/len(train_set),'loss_ts:', total_loss_val/len(test_set))
        pc_best2 = float(PC_mine(labels_pred_ts, labels2_ts))
        if pc_best2 > pc_best: 
            pc_best = pc_best2
            mae_best = float(np.mean(np.abs(labels_pred_ts - labels2_ts)))
            rmse_best = float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2)))
            epsilon_best = float(np.mean(1-np.exp(-((labels_pred_ts - labels2_ts)**2)/(2*(epsi_ts)**2))))           
            
    print(pc_best) 
    print(mae_best) 
    print(rmse_best) 
    print(epsilon_best)   
    
    model_name = opt.model + '_'
    fold_name = str(opt.Fold) + '_'  
    lossfn_name = opt.LossFnc 
      
    
    model_name = models_save_path + model_name + fold_name + lossfn_name +'.pt'  
    torch.save(model.state_dict(), model_name)      
                                                
            
            
    
if __name__ == "__main__":
    Train_FBP()

