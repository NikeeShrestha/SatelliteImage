#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:17:01 2022

@author: mtross
"""

import torch
import numpy as np
import time

ts = time.strftime("%m_%d_%Y__%H")
num_variables = 5
trait = ["TotalGrainMassGrams"]


X_train = torch.load('X_train.pt')
X_test = torch.load('X_test.pt')
y_train = torch.load('y_train.pt')
y_test = torch.load('y_test.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data, labels, pheno_labels):
 
    target = model(data.float())
    
    # Loss autoencoder
    loss_ae = loss_function(target, labels) / X_train.shape[1]

    # Loss Classifier
    classifier_target = regression_model(model.encoder(data.float()))  # Back check the inputs for the encoder part of the model here
    loss_classifier = loss_function_classifier(classifier_target, pheno_labels) / len(y_train)

    optimizer_classifier.zero_grad()
    loss_classifier.backward(retain_graph=True)

    # Total loss both classifier and ae
    if e >= 2000:
        loss = loss_ae + (loss_wt * loss_classifier)
    else:
        loss = loss_ae

    optimizer.zero_grad()

    loss.backward()

    optimizer_classifier.step()
    optimizer.step()


    global train_loss
    global classifier_train_loss

    # Loss for ae only
    global train_loss_ae_only

    classifier_train_loss += loss_classifier.item()
    train_loss += loss.item()
    train_loss_ae_only += loss_ae.item()



class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(kwargs["input_shape"], 2200),
            torch.nn.SELU(),
            torch.nn.Linear(2200, 3000),
            torch.nn.Dropout(0.3),
            torch.nn.SELU(),
            torch.nn.Linear(3000, 2024),
            torch.nn.SELU(),
            torch.nn.Linear(2024, kwargs["output_shape"]))

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(kwargs["output_shape"], 2024),
            torch.nn.SELU(),
            torch.nn.Linear(2024, 3000),
            torch.nn.SELU(),
            torch.nn.Linear(3000, 2200),
            torch.nn.SELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(2200, kwargs["input_shape"]),
            torch.nn.Tanh())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Models
model = AE(input_shape=2151, output_shape=num_variables).to(device)
regression_model = torch.nn.Linear(in_features=num_variables, out_features=1, bias=True).to(device)

loss_function = torch.nn.L1Loss() 

loss_function_classifier = torch.nn.MSELoss()

params_to_optimize = [
    {"params_ae": filter(lambda p: p.requires_grad, model.parameters())},
    {"params_reg": filter(lambda p: p.requires_grad, regression_model.parameters())},
]

optimizer = torch.optim.SGD(params_to_optimize[0]["params_ae"], lr=0.1)
optimizer_classifier = torch.optim.SGD(params_to_optimize[1]["params_reg"], lr=0.1)


history_loss = {
    "classifier_valid_loss": [],
    "classifier_train_loss": [],
    "ae_valid_loss": [],
    "ae_train_loss": [],
}

history_loss_ae_only = {
    "ae_valid_loss_only": [],
    "ae_train_loss_only": [],
}


##################
#### Training ####
##################

# To prevent weird naming issues of output files
if isinstance(trait, list):
    trait = "_".join(trait)

output_mod_name_ae = f"saved_lossWt_model_{trait}_{ts}.pth"
output_mod_name_classifier = f"saved_lossWt_regression_model_{trait}_{ts}.pth"

epochs = 3000
min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    classifier_train_loss = 0.0
    train_loss_ae_only = 0.0

    for data, labels, pheno_labels in zip(X_test[0], X_train[0], y_train[0]):

        if torch.cuda.is_available():
            data, labels, pheno_labels = data.cuda(), labels.cuda(), pheno_labels.cuda()

        if e < 2000:
            regression_model.weight.requires_grad = False
            regression_model.bias.requires_grad = False

            for param in model.parameters():
                param.requires_grad = True

            params_to_optimize = [
                {"params_ae": filter(lambda p: p.requires_grad, model.parameters())},
                {"params_reg": filter(lambda p: p.requires_grad, regression_model.parameters())},
            ]

            loss_wt = 0.0

            train_model(data, labels, pheno_labels)

        if (e >= 2000) and (e < 2300):

            regression_model.weight.requires_grad = True
            regression_model.bias.requires_grad = True

            for param in model.parameters():
                param.requires_grad = False

            params_to_optimize = [
                {"params_ae": filter(lambda p: p.requires_grad, model.parameters())},
                {"params_reg": filter(lambda p: p.requires_grad, regression_model.parameters())
                },
            ]
            loss_wt = 1.0

            train_model(data, labels, pheno_labels)

        if e >= 2300:

            regression_model.weight.requires_grad = True
            regression_model.bias.requires_grad = True

            for param in model.parameters():
                param.requires_grad = True

            params_to_optimize = [
                {"params_ae": filter(lambda p: p.requires_grad, model.parameters())},  # filter is applying lambda function to all the parameters in model.parameters
                {"params_reg": filter(lambda p: p.requires_grad, regression_model.parameters())},
            ]
            
            loss_wt = 0.01  # Seems to work better
            train_model(data, labels, pheno_labels)

    ###################
    ### Validation ####
    ###################

    valid_loss = 0.0
    classifier_valid_loss = 0.0
    valid_loss_ae_only = 0.0

    med_valid_loss = []

    model.eval()  # Optional when not using Model Specific layer
    for data, labels, pheno_labels in zip(
        X_test[0], X_test[0], y_test[0]):

        if torch.cuda.is_available():
            data, labels, pheno_labels = data.cuda(), labels.cuda(), pheno_labels.cuda()
        
        target = model(data.float())
      
        loss = loss_function(target, labels)
     
        valid_loss += loss.item()
        valid_loss_ae_only += loss.item()
        
        # Classifier
        classifier_target = regression_model(model.encoder(data.float()))
        loss_classifier = loss_function_classifier(classifier_target, pheno_labels)
        classifier_valid_loss += loss_classifier.item()

        med_valid_loss.append(valid_loss)

    valid_loss = np.median(med_valid_loss)

    print(f"Epoch {e+1} \t\t Training Loss: {train_loss / X_train.shape[1]} \t\t Validation Loss: {valid_loss / X_test.shape[1]}")
    print(f"Epoch {e+1} \t\t Classifier train loss: {classifier_train_loss / X_train.shape[1]} \t\t Classifier Validation loss: {classifier_valid_loss / X_train.shape[1]}")

    # Add the losses
    history_loss["classifier_valid_loss"].append(classifier_valid_loss)
    history_loss["classifier_train_loss"].append(classifier_train_loss)
    history_loss["ae_valid_loss"].append(valid_loss)
    history_loss["ae_train_loss"].append(train_loss)

    history_loss_ae_only["ae_train_loss_only"].append(train_loss_ae_only)
    history_loss_ae_only["ae_valid_loss_only"].append(valid_loss_ae_only)

    if min_valid_loss > valid_loss:
        print(f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model")
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), output_mod_name_ae)
        torch.save(regression_model.state_dict(), output_mod_name_classifier)

   
