# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 09:33:40 2021

@author: FaresBougourzi
"""

# Face Beauty Prediction Project
# Bougourzi Fares

import numpy as np
import cv2
import torch



"""
Loss Functions

""" 
    
def dy_huber_loss(inputs, targets, beta):
    """
    Dynamic Huber loss function
    
    """
    n = torch.abs(inputs - targets)
    cond = n <= beta
    loss = torch.where(cond, 0.5 * n ** 2, beta*n - 0.5 * beta**2)

    return loss.mean()

def dy_smooth_l1_loss(inputs, targets, beta):
    """
    Dynamic ParamSmoothL1 loss function
    
    """
    n = torch.abs(inputs - targets)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2, n + 0.5 * beta**2 - beta)

    return loss.mean()

def dy_tukey_loss(input, target, c):
    """
    Dynamic Tukey loss function
    
    """    
        
    n = torch.abs(input - target)
    cond = n <= c
    loss = torch.where(cond, ((c** 2)/6) * (1- (1 - (n /c)**2) **3 )  , torch.tensor((c** 2)/6).to('cuda'))

    return loss.mean()


"""
Evaluation Calculations

"""

def MAE_distance(preds, labels):
    return torch.sum(torch.abs(preds - labels))

def PC_mine(preds, labels):
    dem = np.sum((preds - np.mean(preds))*(labels - np.mean(labels)))
    mina = (np.sqrt(np.sum((preds - np.mean(preds))**2)))*(np.sqrt(np.sum((labels - np.mean(labels))**2)))
    return dem/mina 


"""
Face Preprocessing

"""

def Face_align_dt_land(img, vec, dsize):
    # eyes center
    left_eye = vec[59]
    right_eye = vec[58]
    
    # Rotation Angle
    tg_a = (right_eye[1]-left_eye[1])/(right_eye[0]-left_eye[0])   
    ang = np.arctan(tg_a)
    angle = ang*(180/np.pi)

    
    # Rotate the image
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    
    # Rotate the landmarks
    land_rot = np.empty([86,2])
    for i in range(86):
        land_rot[i] = np.sum(rotation_matrix *  np.transpose(np.append(vec[i],1)), axis=1)
        
    # Rotate the centers 
    left_rot = np.sum(rotation_matrix *  np.transpose(np.append(left_eye,1)), axis=1)
    right_rot = np.sum(rotation_matrix *  np.transpose(np.append(right_eye,1)), axis=1)
    
    # Box Boundaries 
    min_eye_y = np.min([left_rot[1],right_rot[1]])
    min_x = np.min(land_rot[0:,0])
    max_x = np.max(land_rot[0:,0])
    max_y = np.max(land_rot[0:,1])

    
    Dist = max_y - min_eye_y
    min_y = min_eye_y - 0.6 * Dist
    if min_y<0:
        min_y =0

    img_cropped = img_rotation[int(np.round(min_y)):int(np.round(max_y)), int(np.round(min_x)):int(np.round(max_x))]

    img_res = cv2.resize(img_cropped, dsize, interpolation = cv2.INTER_AREA)
    return img_res
