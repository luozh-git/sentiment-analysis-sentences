"""
Input: training dataset and pretrained glove embeddings
Ouput: vocabulary and embedding matrix built from the training dataset
"""

import os
import sys
import re
import pickle
import argparse
import pandas as pd
import numpy as np

def read_embeddings(embeddings_file):
	"""
	Read glove or other pretrained embeddings into a dict
	"""
	with open(embeddings_file, encoding = 'utf-8') as f:
		embeddings_index = {}
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:])
			embeddings_index[word] = vector
	
	return embeddings_index

def build_vocabulary_and_embedding_matrix(sentence_list, embeddings_index):
	"""
	sentence_list: a list of sentences
	embeddings_index: dic[word] = word_vector, dict of pretrained embeddings
	embedding_matrix: dict[word_index] = word_vector
	vocabulary: dict[word] = word_index inverse_vocabulary: list[word_index] = word
	"""
	# <unk> is only a placeholder for the zero embedding
	vocabulary = {'<unk>': 0}
	inverse_vocabulary = ['<unk>']
	words_with_no_pretrained_embedding = []

	for sentence in sentence_list:
		# sentences are already cleaned properly
		for word in sentence.split():

			# Discard words without pretrained embeddings
			if word not in embeddings_index:
				words_with_no_pretrained_embedding.append(word)
				continue

			# Starts from len(inverse_vocabulary)=1
			if word not in vocabulary:
				vocabulary[word] = len(inverse_vocabulary)
				inverse_vocabulary.append(word)
	
	EMBEDDING_DIM = 300
	embedding_matrix = np.random.randn(len(vocabulary), EMBEDDING_DIM)
	embedding_matrix[0] = 0  # zero vector for padding
	
	for word, index in vocabulary.items():
		if word == '<unk>': continue

		if word in embeddings_index:
			embedding_matrix[index] = embeddings_index[word]
		else:
			print('{} not found!'.format(word))
	
	return vocabulary, inverse_vocabulary, embedding_matrix, words_with_no_pretrained_embedding


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("train_dateset", help="filename of the cleaned training data in pickle", type=str)
	parser.add_argument("glove_embedding_file", help="filename of the input glove embedding file", type=str)
	args = parser.parse_args()

	if not os.path.isfile(args.train_dateset):
		sys.exit('cannot find file {}'.format(args.train_dateset))
	if not os.path.isfile(args.glove_embedding_file):
		sys.exit('cannot find file {}'.format(args.glove_embedding_file))

	with open(args.train_dateset, 'rb') as f:
		train_df = pickle.load(f)
	
	embeddings_index = read_embeddings(args.glove_embedding_file)

	vocabulary, inverse_vocabulary, embedding_matrix, _ = \
		build_vocabulary_and_embedding_matrix(train_df['sentence_cleaned'], embeddings_index)
	
	# Pickle resutls
	if not os.path.isdir('pickles'):
		os.mkdir('pickles')

	os.chdir('pickles')
	with open('vocabulary.pickle', 'wb') as f: pickle.dump(vocabulary, f)
	with open('inverse_vocabulary.pickle', 'wb') as f: pickle.dump(inverse_vocabulary, f)
	with open('embedding_matrix.pickle', 'wb') as f: pickle.dump(embedding_matrix, f)

if __name__ == '__main__':
	main()
