#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:27:26 2018

@author: rowantseng
"""

import torch.nn as nn
from torchvision import models


class FineTuneModel(nn.Module):
    def __init__(self, modelname, num_classes):
        super(FineTuneModel, self).__init__()
        
        self.modelname = modelname
        if self.modelname == 'alexnet':
            self.features = models.alexnet(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        elif self.modelname == 'resnet50':
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )

        elif self.modelname == 'resnet101':
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )

        elif self.modelname == 'resnet152':
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(models.resnet152(pretrained=True).children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )
    
        elif self.modelname == 'vgg16_bn':
            self.features = models.vgg16_bn(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            
        elif self.modelname == 'vgg19_bn':
            self.features = models.vgg19_bn(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        
        elif self.modelname == 'densenet121':
            self.features = models.densenet121(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Linear(1024, num_classes),
            )

        elif self.modelname == 'densenet169':
            self.features = models.densenet169(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Linear(1664, num_classes),
            )

        elif self.modelname == 'densenet201':
            self.features = models.densenet201(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Linear(1920, num_classes),
            )
            
        elif self.modelname == 'densenet161':
            self.features = models.densenet161(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Linear(2208, num_classes),
            )
        elif self.modelname == 'inception_v3':
            self.inceptionV3 = models.inception_v3(pretrained=True, transform_input=False)
            self.inceptionV3.fc = nn.Linear(2048, num_classes)
                      
            
        else :
            raise('Finetuning not supported on this architecture yet')

        if not self.modelname == 'inception_v3':
            # Freeze weights
            for p in self.features.parameters():
                # p.requires_grad = False
                p.requires_grad = True
            
            # Update weights
            for p in self.classifier.parameters():
                p.requires_grad = True
        else:
            # Freeze weights
            for p in self.inceptionV3.parameters():
                p.requires_grad = True
            
            # Update weights
            for p in self.inceptionV3.fc.parameters():
                p.requires_grad = True
            
    
    def forward(self, x):
        if not self.modelname == 'inception_v3':
            f = self.features(x)
            f = f.view(f.size(0), -1)
            return self.classifier(f)
     
        if self.training:
            f, _ = self.inceptionV3(x)
            return f

        return self.inceptionV3(x)
