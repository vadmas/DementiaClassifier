import os
import re
import requests
import nltk
from collections import defaultdict
import unicodedata

# takes a long string and cleans it up and converts it into a vector to be extracted
# NOTE: Significant preprocessing was done by sed - make sure to run this script on preprocessed text

# Data structure
# data = [
# 	[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]}, <--single utterance
# 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 	],													  <--List of all utterances made during interview
# 	[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 	],
# ]

# constants
PARSER_MAX_LENGTH = 50


# or equivalently and much more efficiently
control_chars = ''.join(map(unichr, range(0,32) + range(127,160)))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

def split_string_by_words(sen,n):
	tokens = sen.split()
	return [" ".join(tokens[(i)*n:(i+1)*n]) for i in range(len(tokens)/n + 1)]

def remove_control_chars(s):
	return control_char_re.sub('',s)

#Input: Sentence to parse
#Output: Parse tree
def get_stanford_parse(sentence, port = 9000):
	#raw = sentence['raw']
	# We want to iterate through k lines of the file and segment those lines as a session
	#pattern = '[a-zA-Z]*=\\s'
	#re.sub(pattern, '', raw)
	re.sub(r'[^\x00-\x7f]',r'', sentence)
	sentence = remove_control_chars(sentence)
	r = requests.post('http://localhost:' + str(port) + '/?properties={\"annotators\":\"parse\",\"outputFormat\":\"json\"}', data=sentence)
	json_obj = r.json()
	return json_obj['sentences'][0]
# trees = []
# for sentence in sentence:
# 	trees.append(sentence['parse'])
# return trees

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

def _processUtterance(uttr):
	uttr = uttr.decode('utf-8').strip()
	# Remove non ascii
	uttr = re.sub(r'[^\x00-\x7f]',r'', uttr)
	tokens = nltk.word_tokenize(uttr)
	tagged_words = nltk.pos_tag(tokens)
	#Get the frequency of every type
	pos_freq = defaultdict(int)
	for word, wordtype in tagged_words:
		if wordtype not in pos_freq:
			pos_freq[wordtype] = 1
		else:
			pos_freq[wordtype] += 1
	#store the sum of frequencies in the hashmap
	pos_freq['SUM'] = len(tokens)
	pt_list = []
	bd_list = []
	for u in split_string_by_words(uttr, PARSER_MAX_LENGTH):
		if u is not "":
			stan_parse = get_stanford_parse(u)
			pt_list.append(stan_parse["parse"])
			bd_list.append(stan_parse["basic-dependencies"])
	datum = {"pos": tagged_words, "raw": uttr, "token": tokens, "pos_freq":pos_freq, "parse_tree":pt_list, "basic_dependencies":bd_list}
	return datum


# Extract data from optima/dbank directory
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

def parse_book(filepath, k):
	if os.path.exists(filepath):
		parsed_data = []
		for filename in os.listdir(filepath):
			if filename.endswith(".txt"):
				# We want to iterate through k lines of the file and segment those lines as a session
				with open(os.path.join(filepath, filename)) as file:
					session = []
					for i, line in enumerate(file):
						if session and i%k == 0:
							print "Parsing ", file.name + ": line ", i
							session_utterances = [_processUtterance(uttr) for uttr in session if _isValid(uttr)]
							parsed_data.append(session_utterances) # Add session		
							# Clear block
							session = []
						else:
							line = unicode(line, errors='ignore')
							session.append(line)
		return parsed_data
	else:
		raise IOError("File not found: " + filepath + " does not exist")
