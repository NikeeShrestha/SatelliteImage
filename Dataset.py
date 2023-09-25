import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

class satelliteimage_dataset(torch.utils.Dataset):
    def __init(self,image_path, label_path):
       self.imagepath=image_path
       self.all_label= pd.read_csv(label_path)
       self.all_filenames= glob.glob(os.path.join(image_path, '*.png'))
    
    def __len__(self):
        return len(self.all_filenames)
        return len(self.all_label)
    
    def __getitem__(self, idx):
        selected_filename= self.all_filenames[idx]
        imagepil= PIL.Image.open(os.path.join(self.imagepath, selected_filename))

        image=torch.utils.to_tensor_and_normalize(imagepil)
        label =torch.Tensor(self.all_label.loc[selected_filename,:].values)

        sample={'data':image,
                'label':label,
                'image_idx': idx
        }

        return sample




      


