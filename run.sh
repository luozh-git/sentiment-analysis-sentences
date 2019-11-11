#!/bin/bash

python preprocess.py train.json train.pickle
python preprocess.py test.json test.pickle
python preprocess.py dev.json dev.pickle

python build_vocabulary_and_embedding_matrix.py pickles/train.pickle glove.840B.300d.txt
python train_test_save_model.py pickles/train.pickle pickles/test.pickle pickles/dev.pickle pickles/vocabulary.pickle pickles/embedding_matrix.pickle
