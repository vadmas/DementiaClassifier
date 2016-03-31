import os
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

def _isValid(inputString):
	# Line should not contain numbers 
	if(any(char.isdigit() for char in inputString)): return False
	# Line should not be empty 
	if not inputString.strip(): return False
	return True

def _processUtterance(uttr):
	uttr = uttr.strip()
	# print uttr
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
	datum = {"pos": tagged_words, "raw": uttr, "token": tokens, "pos_freq":pos_freq}	
	return datum

# Extract data from optima directory
def parse(filepath):
	if os.path.exists(filepath):
		parsed_data = []
		for filename in os.listdir(filepath):
			if filename.endswith(".txt"):
				with open(os.path.join(filepath, filename)) as file:
					print "Parsing: " + file.name
					session_utterances = [_processUtterance(line.decode('utf-8').strip()) for line in file if _isValid(line)]
					parsed_data.append(session_utterances) # Add session
		return parsed_data
	else:
		raise IOError("File not found: " + filepath + " does not exist")
