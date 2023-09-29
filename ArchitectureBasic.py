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
    def __init__(self, **kwargs):
        super.init()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (5, 5), padding = 1)
        self.activation = nn.SELU()
        self.linear1 = nn.Linear(kwargs['input_shape'], 2200)
        self.linear2 = nn.Linear(2200, 3000)
        self.linear3 = nn.Linear(3000, 2024)
        self.linear4 = nn.Linear(2024, kwargs['output_shape'])
