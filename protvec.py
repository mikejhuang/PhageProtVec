import pandas as pd
import numpy as np
from keras.preprocessing.sequence import skipgrams, pad_sequences, make_sampling_table
from keras.preprocessing.text import hashing_trick
from keras.layers import Embedding, Input, Reshape, Dense, merge
from keras.models import Sequential, Model
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import multiprocessing
import csv


#Load Ehsan Asgari's embeddings
#Source: http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287
#Embedding: https://github.com/ehsanasgari/Deep-Proteomics
ehsanEmbed =  []
with open("protVec_100d_3grams.csv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        ehsanEmbed.append(line[0].split('\t'))
threemers = [vec[0] for vec in ehsanEmbed]
embeddingMat = [[float(n) for n in vec[1:]] for vec in ehsanEmbed]
threemersidx = {} #generate word to index translation dictionary. Use for kmersdict function arguments.
for i, kmer in enumerate(threemers):
    threemersidx[kmer] = i
#Set parameters
vocabsize = len(threemersidx)
window_size = 25
num_cores = multiprocessing.cpu_count() #For parallel computing


# Convert sequences to three lists of non overlapping 3mers
def kmerlists(seq):
    kmer0 = []
    kmer1 = []
    kmer2 = []
    for i in range(0, len(seq) - 2, 3):
        if len(seq[i:i + 3]) == 3:
            kmer0.append(seq[i:i + 3])
        i += 1
        if len(seq[i:i + 3]) == 3:
            kmer1.append(seq[i:i + 3])
        i += 1
        if len(seq[i:i + 3]) == 3:
            kmer2.append(seq[i:i + 3])
    return [kmer0, kmer1, kmer2]


# Same as kmerlists function but outputs an index number assigned to each kmer. Index number is from Asgari's embedding
def kmersindex(seqs, kmersdict):
    kmers = []
    for i in range(len(seqs)):
        kmers.append(kmerlists(seqs[i]))
    kmers = np.array(kmers).flatten().flatten(order='F')
    kmersindex = []
    for seq in kmers:
        temp = []
        for kmer in seq:
            try:
                temp.append(kmersdict[kmer])
            except:
                temp.append(kmersdict['<unk>'])
        kmersindex.append(temp)
    return kmersindex


sampling_table = make_sampling_table(vocabsize)


def generateskipgramshelper(kmersindicies):
    couples, labels = skipgrams(kmersindicies, vocabsize, window_size=window_size, sampling_table=sampling_table)
    if len(couples) == 0:
        couples, labels = skipgrams(kmersindicies, vocabsize, window_size=window_size, sampling_table=sampling_table)
    if len(couples) == 0:
        couples, labels = skipgrams(kmersindicies, vocabsize, window_size=window_size, sampling_table=sampling_table)
    else:
        word_target, word_context = zip(*couples)
        return word_target, word_context, labels


def generateskipgrams(seqs, kmersdict=threemersidx):
    #Generate skipgrams for training keras embedding model with negative sampling technique
    #ARGUMENTS:
    # seqs: list, list of amino acid sequences
    # kmersdict: dict to look up index of kmer on embedding, default: Asgari's embedding index
    kmersidx = kmersindex(seqs, kmersdict)
    return Parallel(n_jobs=num_cores)(delayed(generateskipgramshelper)(kmers) for kmers in kmersidx)

def protvec(kmersdict, seq, embeddingweights):
    #Convert seq to three lists of kmers
    kmerlist = kmerlists(seq)
    kmerlist = [j for i in kmerlist for j in i]
    #Convert center kmers to their vector representations
    kmersvec = [0]*100
    for kmer in kmerlist:
        try:
            kmersvec = np.add(kmersvec,embeddingweights[kmersdict[kmer]])
        except:
            kmersvec = np.add(kmersvec,embeddingweights[kmersdict['<unk>']])
    return kmersvec

def formatprotvecs(protvecs):
    protfeatures = []
    for i in range(100):
        protfeatures.append([vec[i] for vec in protvecs])
    protfeatures = np.array(protfeatures).reshape(len(protvecs),len(protfeatures))
    return protfeatures

def formatprotvecsnormalized(protvecs):
    protfeatures = []
    for i in range(100):
        tempvec = [vec[i] for vec in protvecs]
        mean = np.mean(tempvec)
        var = np.var(tempvec)
        protfeatures.append([(vec[i]-mean)/var for vec in protvecs])
    protfeatures = np.array(protfeatures).reshape(len(protvecs),len(protfeatures))
    return protfeatures

def sequences2protvecsCSV(filename, seqs, kmersdict=threemersidx, embeddingweights=embeddingMat):
    #Convert a list of sequences to protvecs and save protvecs to a csv file
    #ARGUMENTS;
    #filename: string, name of csv file to save to, i.e. "sampleprotvecs.csv"
    #seqs: list, list of amino acid sequences
    #kmersdict: dict to look up index of kmer on embedding, default: Asgari's embedding index
    #embeddingweights: 2D list or np.array, embedding vectors, default: Asgari's embedding vectors

    swissprotvecs = Parallel(n_jobs=num_cores)(delayed(protvec)(kmersdict, seq, embeddingweights) for seq in seqs)
    swissprotvecsdf = pd.DataFrame(formatprotvecs(swissprotvecs))
    swissprotvecsdf.to_csv(filename, index=False)
    return swissprotvecsdf

