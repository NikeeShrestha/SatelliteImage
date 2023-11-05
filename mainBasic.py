import Dataset as dat
import architectureBasic as arch
import numpy as np
import torch
import torch.nn as nn
import rasterio
import matplotlib.pyplot as plt 

dataset = dat.satelliteimage_dataset(image_path = '/home/schnablelab/Documents/SixBndImages/Crawfordsville/Hybrids/CroppedTP2', label_path = 'Data/HipsData_TP2_CrawFordsville_yield.csv')
trainSize = int(dataset.__len__() * 0.8)
testSize = dataset.__len__() - trainSize
trainData, testData = torch.utils.data.random_split(dataset, [trainSize, testSize])
trainLoader=torch.utils.data.DataLoader(dataset=trainData, batch_size=1, shuffle=True)
testLoader=torch.utils.data.DataLoader(dataset=testData, batch_size=1, shuffle=True)


model = arch.architectureBasic()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
numEpochs = 10
totalLoss = []
totalvalloss = []
lossFunction = nn.MSELoss()

for epoch in range(numEpochs):
    
    epochLoss = []

    for batch_idx, (data, label) in enumerate(trainLoader):
       
        
        currModel = model(data)
        # print(currModel.shape)
        print(currModel)
        # print(label.shape)
        # print(label)
        loss = lossFunction(currModel, label)
        optimizer.zero_grad()
        # print(loss)
        epochLoss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    epochMeanLoss = np.mean(epochLoss)
    totalLoss.append(epochMeanLoss)
    # print(epochMeanLoss)


    ##Validation

    with torch.no_grad():
        valLoss = []
        for batch_idx, (data, label) in enumerate(testLoader):
            currModel = model(data)
            loss = lossFunction(currModel, label)
            valLoss.append(loss.item())
        valMeanLoss = np.mean(valLoss)
        totalvalloss.append(valMeanLoss)
        # print(valMeanLoss)

plt.plot(range(numEpochs), totalLoss)
plt.plot(range(numEpochs), totalvalloss)
plt.legend(['Train', 'Validation'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
    
        
        

