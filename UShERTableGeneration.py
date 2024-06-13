import csv
import os
import numpy as np
import argparse
from Bio import SeqIO
from UShERHelperFunctions import *
import json

# Example commands

# Spike Protein
# python3 -i UShERTableGeneration.py --ref referenceSequence.fasta --ush usher_barcodes.csv --reg 21563 25384 --outRef 1

# Entire Genome
# python3 -i UShERTableGeneration.py --ref referenceSequence.fasta --ush usher_barcodes.csv --reg 1 29901 --outRef 1


# Pipeline 0: Preparation

# Initialize parser tools
parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

# Initialize arguments
parser.add_argument('--ref', type=str, required=True) # Initialize reference genome (fasta format)
parser.add_argument('--ush', type=str, required=True) # Initialize set of variants (usher file, csv format)
parser.add_argument('--reg', type=str, nargs='+', required=True) # Indicate region of interest (starting position and ending position from reference genome)
parser.add_argument('--name', type=str, required=False) # Determines name to give to every output file (random 5-number string if nothing included here)
parser.add_argument('--outRef', type=int, required=False) # Determines whether output reference file is generated or not (0 if no, 1 if yes; defaults to 0)
parser.add_argument('--dictChoice', type=int, nargs='+', required=False) # Indicate which dictionaries to generate (0 means no, 1 means yes; 5 numbers, with 0 representing UShER instructions, 1 nucleotide sequence, 2 amino acid sequence, 3 nucleotide table, 4 amino acid table; [0,1,1,0,0] if unspecified)

args = parser.parse_args() 

def setArgument(arg,set):
    '''If argument "arg" is unspecified, argument given "set" value.'''
    if arg == None:
        return set
    return arg

# Pipeline I: Generate fasta of genomes

print("Pipeline I:")

# 1.1 Initialize variables
refFile = args.ref
UShERFile = args.ush
[beg,end] = [int(args.reg[0]), int(args.reg[1])]
outRef = setArgument(args.outRef,0)
name = setArgument(args.name, str(rand_5_dig()))
dictChoice = setArgument(args.dictChoice, [0,1,1,0,0])

print("1.1: Initialized variables")

# Begin output reference file
outputReference = [["Reference File: ", str(refFile)],
                   ["UShER File: ", str(UShERFile)],
                   ["Region of Interest: ", beg, end],
                   ["Name for each output: ", name]]

# 1.2 Read the FASTA files
with open(refFile, "r") as fastaFile:
    reference = list(SeqIO.parse(fastaFile, "fasta"))

with open(UShERFile, mode='r') as file:
    UShER = list(csv.reader(file))

print("1.2: read files ", refFile, " and ", UShERFile)

# CHECKPOINT: PROPERLY INDEXED UShER FILE
refCheck = str(reference[0].seq); instrCHECK = [parse_UShER(i) for i in UShER[0][1:]]
for (wuh,ind,mut) in instrCHECK:
    if wuh != refCheck[ind-1]:
        print("UShER Base: ", wuh,
              "Reference Base: ", refCheck[ind-1],
              "Index (UShER): ", ind)


# 1.3 Format UShER table for mutations within region of interest
indicesOfInterest = [0]
for index in range(len(UShER[0])-1):
    (wuh, pos, mut) = parse_UShER(UShER[0][index+1])
    if pos > beg-1 and pos < end+1:
        indicesOfInterest.append(index+1)

formattedUShER = [[row[index] for index in indicesOfInterest] for row in UShER]

print("1.3: formatted UShER table")

# Pipeline II: Generate nucleotide tables
# Each nucleotide table will be an array, with x-axis representing index and y-axis representing nucleotide (A,C,T,G)

print("Pipeline II:")

# 2.1 Generate dictionaries to store variants' nucleotide tables and nucleotide sequences
nucleotideTableDict = {}; nucleotideSequenceDict = {}

# 2.2 Generate reference sequence and reference nucleotide tables
referenceSequence = str(reference[0].seq)[beg-1:end]
referenceNucleotideTable = nucleotideTableConstructor(referenceSequence)

# 2.3 Complete list of UShER instructions
instructions = formattedUShER[0]; instructionsDict = {}

print("2.1-2.3: initialized relevant dictionaries, strings, and arrays")

empty = []

# 2.4 Generate dictionaries
for row in formattedUShER[1:]:
    # Select UShER instructions for each variant
    variant = row[0]; variantInstructions = [instructions[index+1] for index in range(len(row)-1) if float(row[index+1]) == 1]

    if len(variantInstructions) != 0:
        instructionsDict[variant] = [len(variantInstructions), variantInstructions]
    elif len(variantInstructions) == 0:
        empty.append(variant)

    # Add variant to dictionaries
    nucleotideTableDict[variant] = nucleotideTableUpdate(referenceNucleotideTable,variantInstructions, beg)
    nucleotideSequenceDict[variant] = sequenceFromTableConstructor(nucleotideTableDict[variant])

print("2.4: generated sequence and table dictionaries")


# Pipeline III: generate amino acid tables
# Each amino acid table will be an array, similar to the nucleotide arrays, though with amino acids on the y-axis and codons on the x-axis

print("Pipeline III")

# 3.1 Generate dictionaries to store variants' amino acid tables and amino acid sequences
aminoAcidTableDict = {}; aminoAcidSequenceDict = {}

print("3.2 initialized dictionaries")

# 3.2 Convert nucleotide sequences to amino acid sequences and amino acid tables
for variant in list(nucleotideSequenceDict.keys()):
    aminoAcidSequenceDict[variant] = aminoAcidSequenceConstructor(nucleotideSequenceDict[variant])
    aminoAcidTableDict[variant] = aminoAcidTableConstructor(aminoAcidSequenceDict[variant])

print("3.3 generated dictionaries")

# Pipeline IV: Write out files

print("Pipeline IV:")

# 4.1 Define the directory
directory = name

# Make output directory
if not os.path.exists(directory):
    os.makedirs(directory)

if outRef != 0:
    outputReferencePath = os.path.join(directory, "outputReference.txt")
    with open(outputReferencePath, 'w') as file:
        for item in outputReference:
            line = ', '.join(map(str, item))
            file.write(line + '\n')

print("4.1: Created output directory")

# Function to convert numpy arrays to lists
def convert_numpy_dict_to_list(dictionary):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in dictionary.items()}

# 4.2 Convert the dictionaries with numpy arrays to lists
UShERInstructions = convert_numpy_dict_to_list(instructionsDict)
nucleotideTableDict_list = convert_numpy_dict_to_list(nucleotideTableDict)
aminoAcidTableDict_list = convert_numpy_dict_to_list(aminoAcidTableDict)

# 4.2 Function to write a dictionary to a JSON file
def write_dict_to_json(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)

print("4.2: Converted numpy dictionaries to lists")

# 4.3 Write output reference
if outRef != 0:
    outputReferencePath = os.path.join(directory, "outputReference.txt")
    with open(outputReferencePath, 'w') as file:
        for item in outputReference:
            line = ', '.join(map(str, item))
            file.write(line + '\n')

# 4.3 Save the dictionaries to JSON files
if dictChoice[0] == 1:
    write_dict_to_json(UShERInstructions, os.path.join(directory, 'UShER_instructions.json'))

if dictChoice[1] == 1:
    write_dict_to_json(nucleotideSequenceDict, os.path.join(directory, 'nucleotideSequenceDict.json'))

if dictChoice[2] == 1:
    write_dict_to_json(aminoAcidSequenceDict, os.path.join(directory, 'aminoAcidSequenceDict.json'))

if dictChoice[3] == 1:
    write_dict_to_json(nucleotideTableDict_list, os.path.join(directory, 'nucleotideTableDict.json'))

if dictChoice[4] == 1:
    write_dict_to_json(aminoAcidTableDict_list, os.path.join(directory, 'aminoAcidTableDict.json'))

print("4.3: Output files generated")

