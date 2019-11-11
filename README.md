Answer from Luo
=======================
This is a typical sentiment classification problem for sentences. I elected to use a simple LSTM network, 
pretrained word embeddings. There is no manual feature engineering involved.

## How to run 
Start a new virtual env, and run pip install -r requirements.txt
The run.sh sticks the python scripts together. It runs successfully with the files below in the same dir 
run.sh
dev.json
test.json
train.json
preprocess.py
train_test_save_model.py
build_vocabulary_and_embedding_matrix.py
glove.840B.300d.txt
requirements.txt

The deliverables mainly consists of the 3 scripts below
- preprocess.py 
  input: raw data in a json file with wrong formatted quotations
  output: Cleaned data in a pandas dataframe in pickles folder.

- build_vocabulary_and_embedding_matrix.py
  input: training dataset and pretrained glove embeddings
  ouput: vocabulary and embedding matrix built from the training dataset

- train_test_save_model.py
  train a LSTM network, print out test results and save the model


## Results 
The metrics are saved in ./output.txt
The trained model, vocabulary, and cleaned data are save in the ./pickles folder.

## Preprocessing and model
Minimal preprocessing is done. English contractions, which negates the meaning of a sentence, are expanded.
Afterwards, a few non-alphanumeric characters are removed, such apostrophe, and square brackets. Tokenization
is done by simply splitting by spaces, as the sentences are already cleaned up.

A simple LSTM network with pretrained embedding is used to classify the sentences.
The pretrained glove vectors of size 300 are used for the embedding layers of the network. Tokens which 
are not seen in the glove vectors are discarded. The embedding layer is not trainable.

Epoch: 5
batch size: 5
dropout: 0.2
loss: binary_crossentropy
optimizer: adam
activation: sigmoid

## Deploy into production
The trained model can be easily deployed as a web API or application for prediction, 
with the pickled vocabulary, embedding matrix and a few lines of codes for preprocessing, 
tokenization, padding and prediction from the 3 scritps above.

## Libraries used
$ cat requirements.txt 
pandas==0.24.2
scikit-learn==0.21.2
Keras==2.2.5
keras-metrics==1.1.0
Keras-Preprocessing==1.1.0
tensorflow==1.14.0
tensorflow-estimator==1.14.0

And os,pickle,argparse,re,sys
