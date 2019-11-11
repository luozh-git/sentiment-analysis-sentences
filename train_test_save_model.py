"""
train a LSTM network, print out test results and save the model
"""

import os
import pickle
import argparse

import pandas as pd
import numpy as np 

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

import keras_metrics

def tokenize_a_sentence(sentence, vocabulary):        
	sentence2num = []
	for word in sentence.split():
		if word in vocabulary:
			sentence2num.append(vocabulary[word])

	return sentence2num

def build_lstm_model(embedding_matrix, EMBEDDING_DIM = 300, max_length=60):
	"""
	build a simple lstm network
	"""
	model = Sequential()
	model.add(Embedding(input_dim = embedding_matrix.shape[0], output_dim=EMBEDDING_DIM,
                    embeddings_initializer = Constant(embedding_matrix),
                    input_length = max_length,
                    trainable = False))

	model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))

	model.add(Dense(1, activation = 'sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
	return model

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("train_dateset", help="filename of the cleaned training data in pickle", type=str)
	parser.add_argument("test_dateset", help="filename of the cleaned test data in pickle", type=str)
	parser.add_argument("validation_dateset", help="filename of the cleaned validation/dev data in pickle", type=str)
	parser.add_argument("vocabulary", help="filename of the vocabulary", type=str)
	parser.add_argument("embedding_matrix", help="filename of the embedding_matrix", type=str)
	args = parser.parse_args()

	# Read cleand train, test, validation datasets
	# Read vocabulary and embedding matrix prepared 
	with open(args.train_dateset, 'rb') as f: train_df = pickle.load(f)
	with open(args.test_dateset, 'rb') as f: test_df = pickle.load(f)
	with open(args.validation_dateset, 'rb') as f: validation_df = pickle.load(f)
	with open(args.vocabulary, 'rb') as f: vocabulary = pickle.load(f)
	with open(args.embedding_matrix, 'rb') as f: embedding_matrix = pickle.load(f)

	# Tokenize
	for df in [ train_df, validation_df, test_df ]:
		df['sentence2num'] = df['sentence_cleaned'].apply(lambda sentence: tokenize_a_sentence(sentence, vocabulary=vocabulary))

	X_train = list(train_df['sentence2num'])
	y_train = list(train_df['label'])

	X_test = list(test_df['sentence2num'])
	y_test = list(test_df['label'])

	X_val = list(validation_df['sentence2num'])
	y_val = list(validation_df['label'])

	max_length = max([ len(sentence) for sentence in X_train + X_test + X_val ])
	X_train_pad = pad_sequences(X_train, maxlen=max_length, padding='pre')
	X_test_pad = pad_sequences(X_test, maxlen=max_length, padding='pre')
	X_val_pad = pad_sequences(X_val, maxlen=max_length, padding='pre')

	model = build_lstm_model(embedding_matrix, EMBEDDING_DIM = 300, max_length=max_length)

	lstm_trained = model.fit(X_train_pad, y_train, batch_size=64, epochs=5, validation_data=(X_val_pad, y_val), verbose=2)

	output_file = open("output.txt", "w+")

	output_file.write('LSTM network with dropout=0.2\n')
	output_file.write('Number of epochs: 5\n')
	output_file.write('Batch zie: 5\n')

	output_file.write('\nMetrics for training set\n')
	output_file.write('Precision: {}\n'.format(lstm_trained.history['precision'][-1]))
	output_file.write('Recall: {}\n'.format(lstm_trained.history['recall'][-1]))

	output_file.write('\nMetrics for validation set\n')
	output_file.write('Precision: {}\n'.format(lstm_trained.history['val_precision'][-1]))
	output_file.write('Recall: {}\n'.format(lstm_trained.history['val_recall'][-1]))

	y_pred = model.predict(X_test_pad, batch_size=64, verbose=1)
	y_pred_bool = np.round(y_pred).flatten()

	output_file.write('\nMetrics for test set\n')
	output_file.write(classification_report(y_test, y_pred_bool))
	output_file.close()

	# Pickle the trained model 
	if not os.path.isdir('pickles'):
		os.mkdir('pickles')

	os.chdir('pickles')
	with open('model_trained.pickle', 'wb') as f:
		pickle.dump(lstm_trained, f)

if __name__ == '__main__':
	main()
