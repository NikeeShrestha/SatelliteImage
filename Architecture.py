import torch
import torch. nn as nn
import sys
import numpy as np


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
    #    print(self.all_filenames)
    #    self.all_filenames=np.array(self.all_filenames)
    
    def __len__(self):
        return len(self.all_filenames)
        return len(self.all_label)
    
    def __getitem__(self, idx):
        selected_filename= self.all_filenames[idx].split("/")[-1]
        # print(selected_filename)
        imagepil= PIL.Image.open(os.path.join(self.imagepath, selected_filename))

        transform = transforms.Compose([
            transforms.Resize((27, 12)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std as needed
        ])
        image=transform(imagepil) ## model learns better ## we want all the data to be on same scale
        label =torch.Tensor(self.all_label.loc[selected_filename,:].values)

        return image, label

# myloader = satelliteimage_dataset()
dataset = satelliteimage_dataset(image_path="imagedata", label_path="labelfolder/label.csv")
train_set, test_set= torch.utils.data.random_split(dataset, [8,2])
train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=True)

print(train_loader)
print(test_loader)
# sys.exit()
# print(img)
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
        # print(decodelinear4.shape)
        
        return flatten, decodelinear4
    # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5))


# model = architecture(x=img,input_shape=4000, output_shape=10)
# LV=model(x=img)
# print(LV)
# model.summary()


# # print(model[0])
# # print(model[1])

# for batch_idx, (data, label) in enumerate(train_loader):
#         LV=model(data)
#         print(LV)

# optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

# Lossfunction= torch.nn.L1Loss()

num_epochs=10

model=architecture(input_shape=4000, output_shape=10)
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

total_loss=[]
for epoch in range(num_epochs):
    loss_epoch=[]
    for batch_idx, (data, label) in enumerate(train_loader):
        LV=model(data)

        optimizer.zero_grad()

        lossfunction= torch.nn.L1Loss()

        loss=lossfunction(LV[0], LV[1])

        loss_epoch.append(loss.item())

        # print(loss.item())

        loss.backward()

        optimizer.step()

    print(np.mean(loss_epoch))
    total_loss.append(np.mean(loss_epoch))

plt.plot(total_loss)
plt.show()

   
        



