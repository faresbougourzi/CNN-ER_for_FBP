#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:56:05 2021

@author: bougourzi
"""
# Face Beauty Prediction Project
# Bougourzi Fares

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision

class REXINCET(nn.Module):
  def __init__(self):
    super(REXINCET, self).__init__()
    self.model1 = torchvision.models.resnext50_32x4d(pretrained=True)
    self.model1.fc = nn.Linear(2048, 1024)

    self.model2 = torchvision.models.inception_v3(pretrained=True)
    self.model2.fc = nn.Linear(2048, 1024)
    self.fc2 = nn.Linear(2048, 1)

  def forward(self, input1, input2):
    c = self.model1(input1)
    f = self.model2(input2)
    if self.training:
        l, ll = f
    else: 
        l = f
    combined = torch.cat((c,l), dim=1)

    out = self.fc2(F.relu(combined))
    return out
