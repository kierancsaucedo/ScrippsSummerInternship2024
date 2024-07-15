import torch
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from UShERHelperFunctions import *
import argparse
import csv
import os
import random

# Example commands

# Spike Protein (Nucleotide - Constant Noise)
# python3 -i UShERAutoencoder.py --trainPath '97234/observation_*' --evalPath '29746/observation_*' 


# Initialize parser tools
parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

# Initialize arguments
parser.add_argument('--trainPath', type=str, required=True) # path to data to train model
parser.add_argument('--evalPath', type=str, required=False) # path to data to evaluate model
parser.add_argument('--funnel', type=int, nargs='+', required=False) # information on how to encode and decode data
parser.add_argument('--epochs', type=int, required=False) # number of epochs
parser.add_argument('--batchSize', type=int, required=False) # number of observations per batch
parser.add_argument('--learningRate', type=float, required=False) # learning rate of model
parser.add_argument('--weightDecay', type=float, required=False) # weight decay of model
parser.add_argument('--outRef', type=float, required=False) # weight decay of model
parser.add_argument('--patience', type=float, required=False) # variable that effects stop conditions (greater means more training; lesser means less)

args = parser.parse_args() 

def setArgument(arg,set):
    '''If argument "arg" is unspecified, argument given "set" value.'''
    if arg == None:
        return set
    return arg


# Initialize variables

trainPath = args.trainPath
evalPath = setArgument(args.evalPath, trainPath)
funnel = setArgument(args.funnel,[15288,512])
epochs = setArgument(args.epochs,20)
batchSize = setArgument(args.batchSize,32)
learningRate = setArgument(args.learningRate, 1e-5) # to 1e-8
weightDecay = setArgument(args.weightDecay,1e-8)
outRef = setArgument(args.outRef,0)
patience = setArgument(args.patience,10)

outputReference = [["Train Path: ", trainPath],
                   ["Evaluation Path: ", evalPath],
                   ["Funnel: ", funnel],
                   ["Number Epochs: ", epochs],
                   ["Batch Size: ", batchSize],
                   ["Learning Rate: ", learningRate],
                   ["Weight Decay: ", weightDecay],
                   ["Patience", patience]
                   ]

# Helper Functions

def loadDataset(path,batchSize, shuffle):
    '''Loads dataset to train model on.'''
    file_paths = glob.glob(path)
    
    # Transforms input CSV files to tensors within a list
    matrices = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None)
        tensor = torch.tensor(df.values, dtype=torch.float32)
        matrices.append(tensor)

    (rows,columns) = (len(matrices[0]), len(matrices[0][0]))
    
    # Transforms matrix into tensor
    data_tensor = torch.cat(matrices, dim=0).reshape(-1, rows * columns)

    # Generates tensor datset
    dataset = TensorDataset(data_tensor)

    # Loads tensor dataset into batches
    loader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=shuffle)

    return [matrices, rows, columns, data_tensor, dataset, loader]

def MSE_stable_original(mse_values, patience):
    if len(mse_values)<2*patience+1:
        return False
    else:
        recent = mse_values[-patience:]
        prior = mse_values[-2*patience:-patience]
        return 1.1*np.mean(recent) >= np.mean(prior)
        # return np.std(recent)/np.mean(recent) < 0.13

def MSE_stable(mse_values, patience):
    if len(mse_values)<patience:
        return False
    else:
        mse_differences = []
        for index in range(len(mse_values)-1):
            mse_differences.append(mse_values[index] - mse_values[index+1])
        
        if max(mse_differences) > 100*abs(mse_differences[-1]):
            return True
        else:
            return False

def sumIt(list):
    sum=0; ft=3.1; lt=[]
    for i in list:
        if type(i[0].tolist()) == type(ft):
            sum += len(i)
        elif type(i[0].tolist()) == type(lt):
            for j in i:
                sum += len(j)
        else:
            print("Error")
    return sum

# Sanity check with funnel equal to dimension of input

class AE(torch.nn.Module):
    def __init__(self, funnel):
        super(AE, self).__init__()
        
        # Create the encoder part of the autoencoder
        encoder_layers = []
        for i in range(len(funnel)-1):
            encoder_layers.append(torch.nn.Linear(funnel[i], funnel[i+1]))
            encoder_layers.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        # Create the decoder part of the autoencoder
        decoder_layers = []
        for i in range(len(funnel)-1,0,-1):
            decoder_layers.append(torch.nn.Linear(funnel[i], funnel[i-1]))
            decoder_layers.append(torch.nn.ReLU())
        decoder_layers[-1] = torch.nn.Sigmoid()  # Replace the last ReLU with Sigmoid
        self.decoder = torch.nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def trainAE(loader, epochs, rows, columns, loss_function, optimizer, patience):
    outputs = []; losses = []

    for epoch in range(epochs):
        for (data,) in loader:
            data = data.reshape(-1, rows*columns) # Reconstruct tensor matrix object into single list
            reconstructed = model(data) # Run through autoencoder
            loss = loss_function(reconstructed, data) # Determine difference between observation and reconstruction
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        outputs.append((epochs, data, reconstructed))

        if MSE_stable(losses,patience):
            return [outputs, losses]

    return [outputs, losses]

def evalAE(loader, rows, columns, loss_function):
    outputs = []; losses = []; customLosses = []
    for (data,) in loader:
        # Generate outputs and losses additions
        data = data.reshape(-1, rows*columns)
        reconstructed = model(data)
        loss = loss_function(reconstructed, data)
        losses.append(loss.item())
        outputs.append([reconstructed, data])
        # Generate customLosses additions
        uniqueDataIndices = [i for i in range(len(data[0])) if data[0][i]<0.9 and data[0][i]>0.1]
        dataSub = [data[0][i].tolist() for i in uniqueDataIndices]
        reconSub = [reconstructed[0][i].tolist() for i in uniqueDataIndices]
        if uniqueDataIndices != []:
            MSE = np.square(np.subtract(reconSub,dataSub)).mean()
            customLosses.append(MSE)
    return [outputs, losses, customLosses]

def error(Eoutput):
    ori = Eoutput[1][0].tolist(); rec = Eoutput[0][0].tolist(); count = 0; comp = []
    for index in range(len(ori)):
        if round(ori[index],1) != round(rec[index],1):
            count += 1
            comp.append([ori[index],rec[index]])
    return [count, comp]

def allErrorCounter(Eoutputs):
    output = []; num = 0
    for err in Eoutputs:
        ori = err[1][0].tolist(); rec = err[0][0].tolist(); counter = 0
        for index in range(len(ori)):
            if ori[index] != rec[index]:
                counter += 1
        output.append(counter); num += counter
    return [num, output]

def allError(Eoutputs):
    output = []
    for err in Eoutputs:
        output.append(error(err))
    return output


# Train autoencoder model

print("Beginning Pipeline")

[Tmatrices, Trows, Tcolumns, Tdata_tensor, Tdataset, Tloader] = loadDataset(trainPath,batchSize,True)
print("Finished Loading Dataset. Initializing Model.")

print(funnel)

model = AE(funnel)
print("Finished Initializing Model. Training Model.")

[Toutputs, Tlosses] = trainAE(Tloader, epochs, Trows, Tcolumns, torch.nn.MSELoss(), torch.optim.Adam(model.parameters(),lr=learningRate, weight_decay=weightDecay),patience)
print("Finished Training Model. Evaluating Model.")

# Evaluate autoencoder model

[Ematrices, Erows, Ecolumns, Edata_tensor, Edataset, Eloader] = loadDataset(evalPath,1,False)
[Eoutputs, Elosses, EcustomLosses] = evalAE(Eloader, Erows, Ecolumns, torch.nn.MSELoss())
print("Finished Evaluating Model. Producing Outputs.")

# Error Evaluating
errors = allError(Eoutputs)

# 2.1 Define the directory
directory = str(rand_5_dig())

# 2.1 Make output directory
if not os.path.exists(directory):
    os.makedirs(directory)

# 2.2 Write errors.csv to output directory
outputReferencePath = os.path.join(directory, "errors.csv")
with open(outputReferencePath, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(errors)
print("Produced Error Output.")

# Parameter Outputs

if outRef != 0:
    outputReferencePath = os.path.join(directory, "outputReference.txt")
    with open(outputReferencePath, 'w') as file:
        for item in outputReference:
            line = ', '.join(map(str, item))
            file.write(line + '\n')

# Density Plots

def densityPlot(name, directory, xLabel, yLabel, Title, Data, Bins):
    plt.hist(Data, bins=Bins, edgecolor='black')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(Title)
    plt.grid(True)
    outputReferencePath = os.path.join(directory, name)
    plt.savefig(outputReferencePath)
    plt.close()

xLabel = 'Values'
yLabel = 'Frequency'
Bins = 1000
amount = 3

# Assuming Eoutputs is a list of tuples/lists with reconstructed and original data

indices = random.sample(range(len(Eoutputs)), amount)
for index in indices:
    reconstructed, original = Eoutputs[index]
    reconstructed = reconstructed.tolist()[0]; original = original.tolist()[0]
    nameO = str(index) + 'originalData.png'
    titleO = 'Original Data Distribution'
    nameR = str(index) + 'reconstructedData.png'
    titleR = 'Reconstructed Data Distribution'
    densityPlot(nameO, directory, xLabel, yLabel, titleO, original, Bins)
    densityPlot(nameR, directory, xLabel, yLabel, titleR, reconstructed, Bins)
print("Produced Original and Reconstructed Density Plots.")

# Plot losses
xLabel = 'Training Loss (MSE)'
yLabel = 'Frequency'
titleL = 'Loss Metric Distribution'

# Learning Curve Plot
xAxis = list(range(len(Tlosses)))
plt.plot(xAxis, Tlosses, marker='o', linestyle='-', color='b')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.grid(True)
outputReferencePath = os.path.join(directory, 'trainingLoss.png')
plt.savefig(outputReferencePath)
plt.show()
plt.close()  # Close the plot to free memory
print("Produced Learning Curve Plot.")

# Evaluation Plot
plt.boxplot(Elosses)
plt.ylabel('MSE Score')
plt.title('Evaluation Losses')
plt.grid(True)
outputReferencePath = os.path.join(directory, 'evaluationLoss.png')
plt.savefig(outputReferencePath)
plt.close()  # Close the plot to free memory
print("Produced Evaluation Plot.")

# Learning Curve Plot
plt.boxplot(EcustomLosses)
plt.ylabel('MSE Score')
plt.title('Custom Evaluation Losses')
plt.grid(True)
outputReferencePath = os.path.join(directory, 'evaluationCustomLoss.png')
plt.savefig(outputReferencePath)
plt.close()  # Close the plot to free memory
print("Produced Custom Evaluation Plot.")
