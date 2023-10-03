import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import PIL

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
# print(img['label'])
# print(img['idx'])

      


