#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:31:26 2021

@author: bougourzi
"""

# Face Beauty Prediction Project
# Bougourzi Fares

from torchvision.datasets import MNIST
#import warnings
import PIL 
import os
import os.path
import numpy as np
import torch
import cv2

    
    
class Beauty_Db(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  
        self.data, self.targets, self.sigma = torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, sigma = self.data[index], self.targets[index], self.sigma[index]
        img = np.array(img)
        img = img.transpose((1, 2, 0))
        img = img.astype(np.uint8)
                       
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
                        
            
        return  img, target, sigma

    def __len__(self):
        return len(self.data)
    
    
################################################################################
  
    
class Beauty_Db2im(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data, self.targets, self.sigma = torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, sigma = self.data[index], self.targets[index], self.sigma[index]        

        img = np.array(img)
        img1 = img.transpose((1, 2, 0))
        img2 = img.transpose((1, 2, 0))
        img1 = cv2.resize(img1,(224, 224))
        img2 = cv2.resize(img2,(299, 299))        
      
        img1 = img1.astype(np.uint8)        
        img2 = img2.astype(np.uint8)        
              
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return   img1, img2, target, sigma
