import torch
import torch. nn as nn


import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import PIL

import torch.nn.functional as F

class satelliteimage_dataset(torch.utils.data.Dataset):
    def __init__(self,image_path, label_path):
       self.imagepath=image_path
       assert os.path.exists(self.imagepath) ##if its true nothing happens
       assert os.path.exists(label_path) ##same
       self.all_label= pd.read_csv(label_path,index_col=0)
       self.all_filenames= glob.glob(os.path.join(image_path, '*.PNG'))
    
    def __len__(self):
        return len(self.all_filenames)
        return len(self.all_label)
    
    def __getitem__(self, idx):
        selected_filename= self.all_filenames[idx].split("/")[-1]
        print(idx)
        print(selected_filename)
        imagepil= PIL.Image.open(os.path.join(self.imagepath, selected_filename))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std as needed
        ])

        image=transform(imagepil) ## model learns better ## we want all the data to be on same scale
        label =torch.Tensor(self.all_label.loc[selected_filename,:].values)

        sample={'data':image,
                'label':label,
                'image_idx': idx
        }

        return sample

#myloader = satelliteimage_dataset()
img = satelliteimage_dataset(image_path="/Users/nikeeshrestha/Documents/Satellietimage/SatelliteImage/imagedata", label_path="/Users/nikeeshrestha/Documents/Satellietimage/SatelliteImage/labelfolder/label.csv")
print(img.__getitem__(0)['data'].shape)
# print(img.__getitem__(0)['label'])


class architecture(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), padding=1)
        self.linear1=torch.nn.Linear(kwargs["input_shape"], 2200)
        self.activation=torch.nn.SELU()
        self.linear2=torch.nn.Linear(2200, 3000)
        self.dropout=torch.nn.Dropout(0.3)
        self.linear3=torch.nn.Linear(3000, 2024)
        self.linear4=torch.nn.Linear(2024, kwargs["output_shape"])

        
        self.decodelinear1=torch.nn.Linear(kwargs["output_shape"], 2024)
        self.decodelinear2=torch.nn.Linear(2024, 3000)
        self.decodelinear3=torch.nn.Linear(3000, 2200)
        self.decodelinear4=torch.nn.Linear(2200, kwargs["input_shape"])
        self.tanactivation=torch.nn.Tanh()
        
        
    def forward(self, x):
        conv2D_layer1 = self.conv1(x)
        # # print(conv2D_layer1.shape)
        flatten = conv2D_layer1.view(-1)
        # print(flatten.shape)
        encodelinear1=self.linear1(flatten)
        print(encodelinear1.shape)

        ##encoder
        encodelinear1=self.activation(encodelinear1)

        encodelinear2=self.linear2(encodelinear1)
        encodelinear2=self.activation(encodelinear2)

        encodelinear3=self.linear3(encodelinear2)
        encodelinear3=self.activation(encodelinear3)
        encodelinear3=self.dropout(encodelinear3)

        encodelinear4=self.linear4(encodelinear3)

        ##decoder
        decodelinear1=F.selu(self.decodelinear1(encodelinear4))
        decodelinear2=F.selu(self.decodelinear2(decodelinear1))
        decodelinear3=F.selu(self.decodelinear3(decodelinear2))
        decodelinear4=F.selu(self.decodelinear4(decodelinear3))
        
        return encodelinear4
    # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5))
        
model = architecture(x=img.__getitem__(0)['data'],input_shape=4000, output_shape=10)
LV=model(x=img.__getitem__(0)['data'])
print(LV)
# model.summary()