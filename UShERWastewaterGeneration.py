import random
import csv
import os
import numpy as np
import argparse
from Bio import SeqIO
from UShERHelperFunctions import *
import json

# Example commands

# Spike Protein (Constant Noise)
# python3 -i UShERWastewaterGeneration.py --nDict 92394/nucleotideSequenceDict.json --outRef 1 --numObs 1000

# Spike Protein (Variable Noise)
# python3 -i UShERWastewaterGeneration.py --nDict 92394/nucleotideSequenceDict.json --tNoise 2 --outRef 1 --numObs 1000


# Pipeline 0: Preparation

# Initialize parser tools
parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

# Initialize arguments
parser.add_argument('--tNoise', type=int, required=False) # Indicates whether noise is set (1) or range (2) (1 or 2; 1 if unspecified)
parser.add_argument('--noise', type=float, required=False) # Proportion of noise (1% unless specified)
parser.add_argument('--ranNoise', type=float, nargs='+', required=False) # Range of noise (0.5% to 1.5% unless specified)
parser.add_argument('--nDict', type=str, required=True) # Nucleotide dictionary of variants (json file name)
parser.add_argument('--tDict', type=str, required=False) # Indicates whether dictionary is of nucleotide sequences or nucleotide tables ("sequence" or "table"; "sequence" if unspecified)
parser.add_argument('--ranVar', type=int, nargs='+', required=False) # Range of variants in observations (2-6 unless specified)
parser.add_argument('--numObs', type=int, required=False) # Number of observations (10 is unspecified)
parser.add_argument('--outRef', type=int, required=False) # Determines whether output reference file is generated or not (0 if no, 1 if yes; defaults to 0)
parser.add_argument('--round', type=int, required=False) # What each value in the output array is rounded to (4 if unspecified)
parser.add_argument('--name', type=str, required=False) # Determines name to give to every output file ()


args = parser.parse_args() 

def setArgument(arg,set):
    '''If argument "arg" is unspecified, argument given "set" value.'''
    if arg == None:
        return set
    return arg

# Initialize variables
tNoise = setArgument(args.tNoise,1)
noise = setArgument(args.noise,0.01)
[lnoise,hnoise] = setArgument(args.ranNoise,[0.5,1.5])
nDictName = args.nDict
tDict = setArgument(args.tDict,"sequence")
[lbound,hbound] = setArgument(args.ranVar,[2,6])
numObs = setArgument(args.numObs,10)
outRef = setArgument(args.outRef,0)
name = setArgument(args.name, str(rand_5_dig()))
rounding = setArgument(args.round, 4)

# Begin output reference file
outputReference = [["Type of Noise: ", tNoise],
                   ["Amount of Noise per Column (if Type == 1): ", noise],
                   ["Lower and Upper Bounds of Noise (if Type == 2): ", lnoise, hnoise],
                   ["Input Dictionary JSON File: ", nDictName],
                   ["Type of Dictionary: ", tDict],
                   ["Range of Number of Variants: ", lbound, hbound],
                   ["Number of Observations: ", numObs],
                   ["Name for each output: ", name],
                   ["Rounding Number for Each Wastewater Array Value: ", rounding]]

# Generate nDict
with open(nDictName, 'r') as file:
    nDict = json.load(file)


# Pipeline I: Generate wastewater observations

print("Pipeline I:")

# 1.1 Generate table dictionary
tableDict = {}
if tDict == "sequence":
    for key,value in nDict.items():
        table = nucleotideTableConstructor(value)
        tableDict[key] = table
else:
    for key,value in nDict.items():
        tableDict[key] = np.array(value)

print("1.1: Generated table dictionary")

# 1.2 Generate wastewater observations
wasteWaterObservations = generateWastewaterObservation(numObs, tableDict, lbound, hbound, tNoise, noise, lnoise, hnoise, rounding)

print("1.2: Generated wastewater observations")


# Pipeline II: Write out files

print("Pipeline II:")

# 2.1 Define the directory
directory = name

# 2.1 Make output directory
if not os.path.exists(directory):
    os.makedirs(directory)

print("2.1: Created output directory")

# 2.2 Write output files (outputReference.txt, observationKey.csv, and observation_0.csv...)
if outRef != 0:
    outputReferencePath = os.path.join(directory, "outputReference.txt")
    with open(outputReferencePath, 'w') as file:
        for item in outputReference:
            line = ', '.join(map(str, item))
            file.write(line + '\n')

observationKey = [["index","numVOCs","VOCs","props"]] + [[index, wasteWaterObservations[index][1], wasteWaterObservations[index][2], wasteWaterObservations[index][3],] for index in range(len(wasteWaterObservations))]

observationKeyPath = os.path.join(directory, "observationKey.csv")
with open(observationKeyPath, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(observationKey)

for index in range(len(wasteWaterObservations)):
    observation = wasteWaterObservations[index][0]; observationKeyPath = os.path.join(directory, "observation_" + str(index) + ".csv")
    with open(observationKeyPath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(observation)

print("2.2: Generated output files")
