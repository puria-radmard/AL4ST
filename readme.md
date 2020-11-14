<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:18114bdecf6d0a993160009fe8fb183926f30a39ac7c36fb97ae93245097842b
size 872
=======
# Joint extraction of entities and relations-pytorch+CNNs+LSTM
This project is based on [Deep Active Learning for Named Entity Recognition](https://arxiv.org/abs/1707.05928)  
and [Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme](https://arxiv.org/abs/1707.05928)
    

# Dataset


    NYT-CoType
    
# Requirements

    1)python
    2)pytorch
    3)numpy
    4)pickle

# file

    ./data: data file + word2vec
    ./data.py:proces data file and save as binary file
    ./nodel.py: main model
    ./predict.py: not used in this project
    ./stats.py:statistics the data,not import in this project
    ./utils.py:define classes to keep wordvec
    ./model.pt:the model file
    ./record.tcv:the train record
    ./train.py: train file
    ./conv_net:CNN model

# how to run:

    python word2vec.py
    python data
    python train.py
    
    
>>>>>>> e99de2fd835ca2d65d64fcbe09cf17974d8470a0
