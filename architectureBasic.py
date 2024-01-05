import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import PIL
import torchvision.models as models
import Dataset as dl

class architectureBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels = 6, out_channels = 32, kernel_size = (3, 3), padding = 0)
        self.conv2=nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = 0)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*3*1, 128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,32)
        self.fc4=nn.Linear(32,1)
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(0.2)
        
    def forward(self, x):
        # print(x.shape)
        conv1=self.conv1(x)
        # print(conv1.shape)
        conv1=self.pool(self.activation(conv1))
        # print(conv1.shape)
        conv2=self.conv2(conv1)
        # print(conv2.shape)
        conv2=self.pool(self.activation(conv2))
        # print(conv2.shape)
        flatten=conv2.view(-1)
        # print(flatten.shape)
        linear1=(self.activation(self.fc1(flatten)))
        linear2=self.dropout(self.activation(self.fc2(linear1)))
        linear3=(self.activation(self.fc3(linear2)))
        linear4=self.activation(self.fc4(linear3))
        # print(linear4.shape)

        return linear4
    

class architectureresnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1=nn.Conv2d(in_channels = 6, out_channels = 64,kernel_size=(3,3), padding=(3,3), bias=False)

        ##freezing the parametrs of the model
        # for param in self.resnet.parameters():
        #     param.requires_grad=False

        ##Replacing the final fully connected layer  and adjusting the in features to match the last layer of Resnet
        num_features=self.resnet.fc.in_features

        self.resnet.fc=nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        print(self.resnet.fc.requires_grad_)
    def forward(self, x):
        return self.resnet(x)