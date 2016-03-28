import os
import fnmatch
import nltk

# takes a long string and cleans it up and converts it into a vector to be extracted
# NOTE: Significant preprocessing was done by sed - make sure to run this script on preprocessed text

def parseOptima(filepath):
	if os.path.exists(filepath):
		parsed_data = []
		for subdir, dirs, files in os.walk(filepath):
			for file in fnmatch.filter(files, '*.txt'):
				f  = open(os.path.join(subdir, file), 'r')
				id = file.replace('.txt','')
				sentence = ''
				for line in f.readlines():
					if 'P:' in line:
						sentence += line.replace('P:','').replace('\r\n',' ').replace('\n','')
					# id signals beginning of new experiment, 
					# push sentence and clear
					if id in line:
						tokenized = nltk.word_tokenize(sentence)
						if tokenized:
							parsed_data.append(tokenized)
						sentence = ''

		return parsed_data
	else:
		raise IOError("File not found: " + filepath + " does not exist")


def parseDementiaBank(filepath):
	if os.path.exists(filepath):
		parsed_data = []
		for subdir, dirs, files in os.walk(filepath):
			for file in fnmatch.filter(files, '*.txt'):
				f = open(os.path.join(subdir, file), 'r')
				raw = f.read(); f.close()
				raw = raw.decode('utf-8').strip()
				tokenized = nltk.word_tokenize(raw)
				parsed_data.append(tokenized)
		return parsed_data
	else:
		raise IOError("File not found: " + filepath + " does not exist")

