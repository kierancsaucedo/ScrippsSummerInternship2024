import random
import subprocess
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse
import re

# wasteWater.py

def minimap2(target, query):
    '''Command to execute minimap2 alignment from within Python. Uses subprocess.
    INPUTS: target --> reference genome; query --> genome to align to reference genome
    OUTPUT: '''

    # Run minimap2 to align the sequences
    minimap_command = f"minimap2 -a -x asm5 -Y -k14 -w5 -m20 -t4 -DP {target} {query}"
    minimap_output = subprocess.check_output(minimap_command, shell=True, text=True)

    # Process minimap2 output to extract aligned sequences
    minimap_output_adjusted = []
    for line in minimap_output.strip().split('\n'):
        minimap_output_adjusted.append(line.split('\t'))

    return minimap_output_adjusted

def parse_cigar(cigar):
    '''Command to extract all operations within CIGAR string, returns as list of said operations.
    INPUT: CIGAR String (ex. 5M3D9M3I)
    OUTPUT: List of CIGAR commands (ex. [(5,'M'), (3,'D'), (9,'M'), (3,'I')])'''
    return [(int(length), op) for length, op in re.findall(r'(\d+)([MID])', cigar)]

def apply_cigar(sequence, cigar, offset):
    '''Command to manipulate a sequence according to its CIGAR string.
    INPUT: Sequence (ex. AAAACCCCTTTTGGGG), CIGAR string (ex. 8M3D5M6D3M), and offset (number of bases before alignment begins; ex. 12).'''
    parsed_cigar = parse_cigar(cigar)
    result = ["-" for i in range(offset)]
    seq_index = 0

    for length, op in parsed_cigar:
        if op == 'M':  # Match or mismatch
            result.append(sequence[seq_index:seq_index+length])
            seq_index += length
        elif op == 'D':  # Deletion
            result.append('-' * length)
        elif op == 'I':  # Insertion
            seq_index += length  # Normally you would handle this if you had the other sequence
        elif op == 'S':  # Soft clipping
            seq_index += length

    return ''.join(result)

# Example usage
sequence = "ACTAGAATGGCT"; reference = "CCATACTGAACTGACTAAC"; cigar = "3M1I3M1D5M"; offset = 4
aligned_sequence = apply_cigar(sequence, cigar, offset)

def processMinimap2Alignments(alignments):
    '''This function accepts the minimap2 function output and processes it.
    INPUT: Output of minimap2 (list)
    OUTPUT: dictionary with sequence name as key, best secondary alignment CIGAR score as value
    '''
    outputDict = {}
    for alignment in alignments[3:]:
        ID = alignment[0]; num = alignment[1]; offset = int(alignment[3])-1; CIGAR = alignment[5]; score = int(alignment[4])
        if num == '256' and score == 0 and ID not in list(outputDict.keys()):
            outputDict[ID] = [offset, CIGAR]
    return outputDict


# usher.py

nucleotideDict = {'A':0, 'C':1, 'T':2, 'G':3}

nucleotideDictInv = {0:'A', 1:'C', 2:'T', 3:'G'}

aminoAcidDict = {
    0: 'A',  # Alanine
    1: 'C',  # Cysteine
    2: 'D',  # Aspartic acid
    3: 'E',  # Glutamic acid
    4: 'F',  # Phenylalanine
    5: 'G',  # Glycine
    6: 'H',  # Histidine
    7: 'I',  # Isoleucine
    8: 'K',  # Lysine
    9: 'L',  # Leucine
    10: 'M', # Methionine
    11: 'N', # Asparagine
    12: 'P', # Proline
    13: 'Q', # Glutamine
    14: 'R', # Arginine
    15: 'S', # Serine
    16: 'T', # Threonine
    17: 'V', # Valine
    18: 'W', # Tryptophan
    19: 'Y', # Tyrosine
    20: '*', # Stop
}

aminoAcidDictInv = {v: k for k, v in aminoAcidDict.items()}

def parse_UShER(UShER_string):
    '''Turns UShER string (ex. 'A10645G') into list with first character, index, last character (ex. ['A',10645,'G']).'''
    (org, num, mut) = re.findall(r'([A-Z])(\d+)([A-Z])',UShER_string)[0]
    return (org, int(num), mut)

def nucleotideTableConstructor(sequence):
    '''Turns sequence of nucleotides into a 4xlen(sequence) array, with 1s to represent whether an
    A (first row), C (second row), T (third row), or G (fourth row) is at the corresponding position.'''
    A = []; C = []; T = []; G = []
    for base in sequence:
        if base == 'A':
            A.append(1); C.append(0); T.append(0); G.append(0)
        elif base == 'C':
            A.append(0); C.append(1); T.append(0); G.append(0)
        elif base == 'T':
            A.append(0); C.append(0); T.append(1); G.append(0)
        elif base == 'G':
            A.append(0); C.append(0); T.append(0); G.append(1)
    return np.array([A,C,T,G])

def sequenceFromTableConstructor(table):
    '''The inverse function to "nucleotideTableConstructor".'''
    output = []
    for index in range(len(table[0])):
        for num in range(4):
            if table[num][index] == 1:
                output.append(nucleotideDictInv[num])
    return ''.join(output)

def nucleotideTableUpdate(table, UShERs, offset):
    '''Updates a nucleotide table, taking in changes from a list of UShER-encoded instructions.'''
    output = table.copy()
    for UShER in UShERs:
        (wuh,pos,mut) = parse_UShER(UShER)
        if table[nucleotideDict[wuh]][pos-offset] != 1:
            print("Error at position ", pos, ". Wrong nucleotide.")
        else:
            output[nucleotideDict[wuh]][pos-offset] = 0
            output[nucleotideDict[mut]][pos-offset] = 1
    return output

def aminoAcidSequenceConstructor(DNASequence):
    '''Takes in a DNA sequence (string), outputs a amino acid sequence (string).'''
    return str(Seq(DNASequence).translate())

def aminoAcidTableConstructor(proteinSequence):
    '''Takes in a protein sequence (string), outputs an amino acid table (array).'''
    table = np.zeros((len(aminoAcidDict),len(proteinSequence)))
    for index in range(len(proteinSequence)):
        aminoAcid = proteinSequence[index]; row = aminoAcidDictInv[aminoAcid]
        table[row][index] = 1
    return table

# Pipeline IV of usher.py Helper Functions

def rand_5_dig():
    return random.randint(10000, 99999)

def order(list):
    '''Arranges a list of numbers from least to greatest.'''
    output = []
    for i in range(len(list)):
        low = min(list); output.append(low); list.remove(low)
    return output

def normalize(original, noise):
    '''Accepts two nxm matrices, "original" and "noise".
    Scales each column in "original" so that the sum of that
    column and the sum of the same indexed column in "noise"
    adds up to 1. (It is assumed the sum of each "noise"
    column is between 0 and 1, since that column is not scaled.)
    Returns scaled_original+noise arrays.'''
    
    # Calculate the sum of each column in the original and noise matrices
    original_sums = np.sum(original, axis=0)
    noise_sums = np.sum(noise, axis=0)
    
    # Scale the original matrix columns
    scaling_factors = 1 - noise_sums
    scaled_original = original * (scaling_factors / original_sums)
    
    return scaled_original+noise

def generateNoise(perc, n, m, random_type, lbound=None, hbound=None):
    '''Generates an "n"-rows by "m"-columns array of noise.
    Type One: each column adds up to "perc".
    Type Two: each column adds up to between "lbound" and "hbound"'''

    if random_type not in [1, 2]:
        raise ValueError("random_type must be either 1 or 2")

    noise = np.zeros((n, m))
    
    if random_type == 1:
        # Type One: Each column adds up to "perc"
        for col in range(m):
            noise[:, col] = np.random.random(n)
            noise[:, col] = noise[:, col] / noise[:, col].sum() * perc

    elif random_type == 2:
        if lbound is None or hbound is None:
            raise ValueError("lbound and hbound must be specified for Type Two")
        if lbound > hbound:
            raise ValueError("lbound must be less than or equal to hbound")
        
        # Type Two: Each column adds up to between "lbound" and "hbound"
        for col in range(m):
            col_sum = np.random.uniform(lbound, hbound)
            noise[:, col] = np.random.random(n)
            noise[:, col] = noise[:, col] / noise[:, col].sum() * col_sum
    
    return noise

def generateWastewaterObservation(number, dictionary, lBoundVar, hBoundVar, tNoise, noise, lBoundNoise, hBoundNoise, rounding):
    observations = []

    # Generate observations
    for observation in range(number):
        
        # Choose number of variants
        numVOCs = random.choice(list(range(lBoundVar,hBoundVar+1)))
        
        # Choose variants
        VOCs = random.sample(list(dictionary.keys()),numVOCs)

        # Choose proportion of each variant
        propPartitions = order([random.random() for VOC in range(numVOCs-1)]+[0,1]) # Randomly partitions range between 0 and 1 into numVOCs sections
        props = [propPartitions[i+1]-propPartitions[i] for i in range(numVOCs)] # Determines size of each section from random partition (size section == prop. associated var.)

        # Generate dummy wastewater array
        rows = len(dictionary[VOCs[0]]); columns = len(dictionary[VOCs[0]][0])
        wastewaterObservation = np.zeros((rows,columns))

        # Generate wastewater observation by adding random proportions of each variant
        for index in range(numVOCs):
            VOC = dictionary[VOCs[index]]; proportion = props[index]
            wastewaterObservation += VOC * proportion

        # Generate associated noise
        noise_array = generateNoise(noise, rows, columns, tNoise, lBoundNoise, hBoundNoise)

        # Combine wastewater observation with noise
        wastewaterObservationOutput = np.round(normalize(wastewaterObservation,noise_array), rounding)

        # Add to output list
        observations.append([wastewaterObservationOutput, numVOCs, VOCs, props])

    return observations

def generateFasta(seqName, seq):
    '''Accepts a sequence and sequence name, outputs a fasta (named after the sequence name) containing this sequence.'''
    sequence = [(seqName,seq)]; record = [SeqRecord(Seq(seq), id=seq_id, description="") for seq_id, seq in sequence]
    outputFile = seqName + ".fasta"; SeqIO.write(record, outputFile, "fasta")


def randomFasta(dictionary, number):
    '''Generates "number" fasta genome documents from "dictionary" with variant-name keys and nucleotide-genomes values.'''
    keys = list(dictionary.keys()); nums = list(range(len(keys)))
    for num in range(number):
        ind = random.choice(nums); key = keys[ind]; seq = dictionary[key]
        generateFasta(key,seq)