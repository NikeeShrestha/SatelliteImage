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
        self.conv = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (5, 5), padding = 1)
        self.linear1 = nn.Linear(4000, 2000)
        self.linear2 = nn.Linear(2000, 1000)
        self.linear3 = nn.Linear(1000, 500)
        self.linear4 = nn.Linear(500, 250)
        self.linear5 = nn.Linear(250, 100)
        self.linear6 = nn.Linear(100, 50)
        self.linear7 = nn.Linear(50, 1)

    def forward(self, x):
        conv = self.conv(x)
        flatten = conv.view(-1)
        linear1 = F.relu(self.linear1(flatten))
        linear2 = F.relu(self.linear2(linear1))
        linear3 = F.relu(self.linear3(linear2))
        linear4 = F.relu(self.linear4(linear3))
        linear5 = F.relu(self.linear5(linear4))
        linear6 = F.relu(self.linear6(linear5))
        linear7 = self.linear7(linear6)
        
        return linear7
        