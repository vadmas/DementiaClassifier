import os
import re
import requests
import nltk

# takes a long string and cleans it up and converts it into a vector to be extracted
# NOTE: Significant preprocessing was done by sed - make sure to run this script on preprocessed text

# Data structure
# data = [
# 	[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[]}, <--single utterance
# 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[]},
# 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[]},
# 	],													  <--List of all utterances made during interview
# 	[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[]},
# 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[]},
# 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[]},
# 	],
# ]

def get_parse_tree(sentences, port = 9000):
	#raw = sentence['raw']
	#pattern = '[a-zA-Z]*=\\s'
	#re.sub(pattern, '', raw)
	re.sub(r'[^\x00-\x7f]',r'', sentences)
	r = requests.post('http://localhost:' + str(port) + '/?properties={\"annotators\":\"parse\",\"outputFormat\":\"json\"}', data=sentences)
	json_obj = r.json()
	sentences = json_obj['sentences']
	trees = []
	for sentence in sentences:
		trees.append(sentence['parse'])
	return trees

def _isValid(inputString):

	# Line should not contain numbers 
	if(any(char.isdigit() for char in inputString)): return False
	# Line should not be empty 
	elif not inputString.strip(): return False
	# Line should contain characters (not only consist of punctuation)
	elif not bool(re.search('[a-zA-Z]', inputString)):
		return False
	else:
		return True

import unicodedata, re

# or equivalently and much more efficiently
control_chars = ''.join(map(unichr, range(0,32) + range(127,160)))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

def remove_control_chars(s):
	return control_char_re.sub('',s)

def remove_control_chars(s):
	return control_char_re.sub('', s)

def _processUtterance(uttr):
	uttr = uttr.decode('utf-8').strip()
	# Remove non ascii
	uttr = re.sub(r'[^\x00-\x7f]',r'', uttr)

	tokens = nltk.word_tokenize(uttr)
	tagged_words = nltk.pos_tag(tokens)
	#Get the frequency of every type
	pos_freq = {}
	for word, wordtype in tagged_words:
		if wordtype not in pos_freq:
			pos_freq[wordtype] = 1
		else:
			pos_freq[wordtype] += 1
	#store the sum of frequencies in the hashmap
	pos_freq['SUM'] = len(tokens)
	parse_tree = get_parse_tree(uttr)
	datum = {"pos": tagged_words, "raw": uttr, "token": tokens, "pos_freq":pos_freq, "parse_tree":parse_tree}
	return datum

# Extract data from optima directory


def parse(filepath):
	if os.path.exists(filepath):
		parsed_data = []
		for filename in os.listdir(filepath):
			if filename.endswith(".txt"):
				with open(os.path.join(filepath, filename)) as file:
					print "Parsing: " + file.name
					session_utterances = [_processUtterance(line) for line in file if _isValid(line)]
					parsed_data.append(session_utterances) # Add session
		return parsed_data
	else:
		raise IOError("File not found: " + filepath + " does not exist")
