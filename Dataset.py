import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import PIL
import rasterio

class satelliteimage_dataset(torch.utils.data.Dataset):
    def __init__(self,image_path, label_path):
       self.imagepath=image_path
       assert os.path.exists(self.imagepath) ##if its true nothing happens
       assert os.path.exists(label_path) ##same

       self.all_label= pd.read_csv(label_path,index_col=0)
       self.all_filenames= glob.glob(os.path.join(image_path, '*.TIF'))
    #    print(self.all_filenames)
    #    self.all_filenames=np.array(self.all_filenames)
    
    def __len__(self):
        return len(self.all_filenames)
        return len(self.all_label)
    
    def __getitem__(self, idx):
        selected_filename= self.all_filenames[idx].split("/")[-1]
        # print(selected_filename)
        # imagepil= PIL.Image.open(os.path.join(self.imagepath, selected_filename))

        if selected_filename not in self.all_label.index:
            print(selected_filename, "no data for this file")
            return self.__getitem__((idx + 1) % len(self))
        
        with rasterio.open(os.path.join(self.imagepath, selected_filename), 'r') as src:
            image=src.read()
            # print(image.shape)
            image = image.astype(np.float32)
            # print(image.shape)
            image = image.transpose((1,2,0))
            # image = (image - image.min()) / (image.max() - image.min())

        transform = transforms.Compose([
            # transforms.Resize((27, 12)),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=45),
            transforms.Normalize(mean=[541.864709,708.803880,573.717252,4653.988359,2227.880503,590.645810], std=[106.979793, 83.798613, 66.100552, 358.867302,154.001127,51.439945])  # Adjust mean and std as needed
        ])
        # image=transform(image) ## model learns better ## we want all the data to be on same scale
        # print(image.shape)
        label =torch.Tensor(self.all_label.loc[selected_filename,:].values)
        filename=selected_filename
        # print(label.shape)

        return image, label, filename
# print(img['label'])
# print(img['idx'])

      
# satelliteimage_dataset(image_path = 'Data/', label_path = 'labelfolder/label.csv')
    
class satelliteimage_dataset_getitemfixed(torch.utils.data.Dataset):
    def __init__(self,image_path, label_path):
       self.imagepath=image_path
       assert os.path.exists(self.imagepath) ##if its true nothing happens
       assert os.path.exists(label_path) ##same

       self.all_label= pd.read_csv(label_path,index_col=0)
       self.all_filenames= glob.glob(os.path.join(image_path, '*.TIF'))
    #    print(self.all_filenames)
    #    self.all_filenames=np.array(self.all_filenames)
    
    def __len__(self):
        return len(self.all_filenames)
        # return len(self.all_label)
    
    def __getitem__(self, idx):
        while True:
            selected_filename = self.all_filenames[idx].split("/")[-1]

            if selected_filename in self.all_label.index:
                # Process the file as it has an associated label
                with rasterio.open(os.path.join(self.imagepath, selected_filename), 'r') as src:
                    image = src.read()
                    image = image.astype(np.float32)
                    image = image.transpose((1, 2, 0))

                transform = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomRotation(20),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[541.864709,708.803880,573.717252,4653.988359,2227.880503,590.645810], std=[106.979793, 83.798613, 66.100552, 358.867302,154.001127,51.439945])
                    # Add other transformations as necessary
                ])

                label = torch.Tensor(self.all_label.loc[selected_filename, :].values)
                return transform(image), label, selected_filename
            else:
                # Print a warning and skip to the next file
                # print(f"Warning: {selected_filename} has no data. Skipping.")
                idx = (idx + 1) % len(self)
                if idx == 0:
                    raise RuntimeError("No valid images found in dataset.")
        return image, label, filename


