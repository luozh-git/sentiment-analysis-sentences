"""
input: raw data in a json file with wrong formatted quotations
output: cleaned data in a pandas dataframe
"""
import argparse
import os
import re
import pickle
import pandas as pd
import numpy as np

def read_sentences(filename):
	"""
	Read train.json, dev.json, test.json into a pandas dataframe 
	"""
	with open(filename, 'r') as f:
		lines = f.readlines()
    
	sentences = []
	for line in lines:
		label, sentence = int(line[10]), line[26:-3].strip()
		sentences.append((label, sentence))
    
	return pd.DataFrame(sentences, columns = ['label', 'sentence'])

def expand_contractions(sentence):
	"""
	Expand common contractions with negations, as the contracted
	version are not in the embedding index
	"""
	contractions_dict = {
		"ain't": "are not",
		"aren't": "are not",
		"can't": "can not",
		"couldn't": "could not",
		"didn't": "did not",
		"doesn't": "does not",
		"don't": "do not",
		"hadn't": "had not",
		"hasn't": "has not",
		"haven't": "have not",
		"isn't": "is not",
		"mustn't": "must not",
		"shouldn't": "should not",
		"wasn't": "was not",
		"weren't": "were not",
		"won't": "will not",
		"wouldn't": "would not",
	}
	contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

	def replace(match):
		return contractions_dict[match.group(0)]
    
	return contractions_re.sub(replace, sentence)

def preprocessing(sentence):
	"""
	Minimalistic preprocessing of sentences
	"""
	# Remove square brackets
	sentence = sentence.replace('[', '')
	sentence = sentence.replace(']', '')
   
	# Remove backslashes
	sentence = sentence.replace("\\", "")
	sentence = expand_contractions(sentence)
    
	# After the expansion, remaining apostrophes are kept
	sentence = sentence.replace("'", " ")
	return sentence

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="filename of the input json formatted file", type=str)
	parser.add_argument("output", help="filename of the output pickle file", type=str)
	args = parser.parse_args()
	df = read_sentences(args.input)
	df['sentence_cleaned'] = df['sentence'].apply(preprocessing)
	
	if not os.path.isdir('pickles'):
		os.mkdir('pickles')
	
	os.chdir('pickles')
	with open(args.output, 'wb') as f:
		pickle.dump(df, f)

if __name__ == '__main__':
	main()
