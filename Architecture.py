import torch
import torch. nn as nn

class architecture(torch.nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5))
        
    


        
    def forward(self, x):
        
model = architecture()
model