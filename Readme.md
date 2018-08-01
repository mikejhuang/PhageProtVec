# Protvec: Amino Acid Embedding Representation for Machine Learning Features

## Objectives
1. Extract features from amino acid sequences for machine learning
2. Use features to predict protein family and other structural properties

## Abstract
This project attempts to reproduce the results from [Asgari 2015](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287) and to expand it to phage sequences and their protein families. Currently, Asgari's classification of protein families can be reproduced with his using his [trained embedding.](https://github.com/ehsanasgari/Deep-Proteomics). However, his results cannot be reproduced with current attempts to train using the skip-gram negative sampling method detailed in [this tutorial.](http://adventuresinmachinelearning.com/word2vec-keras-tutorial/) Training samples have been attempted with the SwissProt database. 

## Introduction
Predicting protein function with machine learning methods require informative features that is extracted from data. A natural language processing (NLP) technique, known as Word2Vec is used to represent a word by its context with a vector that encodes for the probability a context would occur for a word. These vectors are effective at representing meanings of words since words with similar meanings would have similar contexts. For example, the word cat and kitten would have similar contexts that they are used in since they have very similar meanings. These words would thus have very similar vectors. 

## Methods
1. Preprocessing
    1. Load dataset containing protein amino acid sequences and Asgari's embedding
    2. [Convert sequences to three lists of non-overlapping 3-mer words](https://www.researchgate.net/profile/Mohammad_Mofrad/publication/283644387/figure/fig4/AS:341292040114179@1458381771303/Protein-sequence-splitting-In-order-to-prepare-the-training-data-each-protein-sequence.png) 
    3. Convert 3-mers to numerical encoding using kmer indicies from Asgari's embedding (row dimension)
    4. Generate skipgrams with [Keras function](https://keras.io/preprocessing/sequence/)  
        Output: [target word, context word](http://mccormickml.com/assets/word2vec/training_data.png), label  
        Label refers to true or false target/context pairing generated for the negative sampling technique             
2. Training embedding
    1. Create negative sampling skipgram model with Keras [using technique from this tutorial](http://adventuresinmachinelearning.com/word2vec-keras-tutorial/)
3. Generate ProtVecs from embedding for a given protein sequence
    1. Break protein sequence to list of kmers
    2. Convert kmers to vectors by taking the dot product of its one hot vector with the embedding 
    3. Sum up all vectors for all kmers for a single vector representation for a protein (length 100)        
4. Classify protein function with ProtVec features (results currently not working, refer to R script)
    1. Use protvecs as training features
    2. Use pfam as labels
    3. For a given pfam classification, perform binary classification with all of its positive samples and randomly sample an equal amount of negative samples
    4. Train SVM model 
    
    
## Resources 
1. Intuition behind Word2Vec http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
2. Tutorial followed for implementation of skip-gram negative sampling (includes code) http://adventuresinmachinelearning.com/word2vec-keras-tutorial/
3. Introduction to protein function prediction
http://biofunctionprediction.org/cafa-targets/Introduction_to_protein_prediction.pdf
