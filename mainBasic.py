import Dataset as dat
import architectureBasic as arch
import numpy as np
import torch
import torch.nn as nn
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

dataset = dat.satelliteimage_dataset(image_path = '/home/schnablelab/Documents/SixBndImages/Crawfordsville/Hybrids/CroppedTP2', label_path = 'Data/HipsData_TP2_CrawFordsville_yield.csv')
trainSize = int(dataset.__len__() * 0.8)
testSize = dataset.__len__() - trainSize
trainData, testData = torch.utils.data.random_split(dataset, [trainSize, testSize])
trainLoader=torch.utils.data.DataLoader(dataset=trainData, batch_size=1, shuffle=True)
testLoader=torch.utils.data.DataLoader(dataset=testData, batch_size=1, shuffle=True)


model = arch.architectureBasic()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
numEpochs = 4000
totalLoss = []
totalvalloss = []
lossFunction = nn.MSELoss()

for epoch in range(numEpochs):
    
    epochLoss = []

    for batch_idx, (data, label,f) in enumerate(trainLoader):
       
        
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
        for batch_idx, (data, label, f) in enumerate(testLoader):
            currModel = model(data)
            loss = lossFunction(currModel, label)
            valLoss.append(loss.item())

        valMeanLoss = np.mean(valLoss)
        totalvalloss.append(valMeanLoss) 
        # print(valMeanLoss)

        Finalprediction=[]

        if epoch == numEpochs-1:
            for batch_idx, (data, label, f) in enumerate(testLoader):
                file=f
                finalModel=model(data).item()
                labelfinal=label.item()
                Finalprediction.append(
                {'prediction': finalModel,
                    'label': labelfinal,
                    'file': file,
                }
            )
        # print(valMeanLoss)

Finalprediction=pd.DataFrame(Finalprediction)
Finalprediction.to_csv('Finalprediction.csv', index=False)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(range(numEpochs), totalLoss)
ax.plot(range(numEpochs), totalvalloss)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend(['Train', 'Validation'])
plt.savefig("Validation.png")

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(Finalprediction['label'], Finalprediction['prediction'])
ax.set_xlabel('RealValue')
ax.set_ylabel('PredictedValue')
r,pval=pearsonr(Finalprediction['label'], Finalprediction['prediction'])
ax.annotate('r = {:.2f}'.format(r), (min(Finalprediction['label']), max(Finalprediction['prediction'])), size=12)
plt.savefig("Validation_values.png")


        
        

