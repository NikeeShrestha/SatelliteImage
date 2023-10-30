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
import Dataset as dl

class architectureBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels = 6, out_channels = 32, kernel_size = (3, 3), padding = 0)
        self.conv2=nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = 0)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*20*10, 128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,32)
        self.fc4=nn.Linear(32,1)
        self.activation=nn.ReLU()
        
    def forward(self, x):
        conv1=self.conv1(x)
        conv1=self.pool(self.activation(conv1))
        conv2=self.conv2(conv1)
        conv2=self.pool(self.activation(conv2))
        flatten=conv2.view(-1)
        linear1=self.activation(self.fc1(flatten))
        linear2=self.activation(self.fc2(linear1))
        linear3=self.activation(self.fc3(linear2))
        linear4=self.activation(self.fc4(linear3))

        return linear4
        