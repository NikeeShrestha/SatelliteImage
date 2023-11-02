import Dataset as dat
import architectureBasic as arch
import numpy as np
import torch
import torch.nn as nn
import rasterio

dataset = dat.satelliteimage_dataset(image_path = 'imagedata/', label_path = 'Data/HipsData_TP2_CrawFordsville_yield.csv')
# print(dataset.__len__())
trainSize = int(dataset.__len__() * 0.8)
# print(trainSize)
testSize = dataset.__len__() - trainSize
trainData, testData = torch.utils.data.random_split(dataset, [trainSize, testSize])
trainLoader=torch.utils.data.DataLoader(dataset=trainData, batch_size=1, shuffle=True)
testLoader=torch.utils.data.DataLoader(dataset=testData, batch_size=1, shuffle=True)

# for batch_idx, (data, label) in enumerate(trainLoader):
#     print(data)
#     print(label)


model = arch.architectureBasic()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
numEpochs = 10
totalLoss = []
lossFunction = nn.MSELoss()

for epoch in range(numEpochs):
    
    epochLoss = []
    
    for batch_idx, (data, label) in enumerate(trainLoader):
        
        currModel = model(data)
        print(currModel)
        optimizer.zero_grad()
        loss = lossFunction(currModel, label)
        # print(loss)
        epochLoss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    epochMeanLoss = np.mean(epochLoss)
    totalLoss.append(epochMeanLoss)
    print(epochMeanLoss)
    
        
        

