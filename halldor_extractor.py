# takes in a list of string and turns them into a list of features 

import nltk
from collections import defaultdict

#input: tokenized string
#returns: dictionary of frequencies for each type of word from the tokenized string
def pos_tag(tokens):

	#get pos tags
	tagged_words = nltk.pos_tag(tokens)
	
	#Get the frequency of every type
	pos_freq = defaultdict()
	for word, wordtype in tagged_words:

		if wordtype not in pos_freq:
			pos_freq[wordtype] = 1
		else:
			pos_freq[wordtype] += 1

	#store the sum of frequencies in the hashmap
	pos_freq['SUM'] = len(tokens)

	return pos_freq

"""
=============================================================

WORD TYPE COUNTS

=============================================================
"""


#input: NLP object for one paragraph
#returns: number of normalized nouns in text
def getNumNouns(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['NN'] + pos_freq['NNP'] + pos_freq['NNS']+ pos_freq['NNPS'])/pos_freq['SUM']


#input: NLP object for one paragraph
#returns: number of normalized verbs in text
def getNumVerbs(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['VB'] + pos_freq['VBD'] + pos_freq['VBG'] + pos_freq['VBN'] + pos_freq['VBP'] + pos_freq['VBZ'])/pos_freq['SUM']


#input: NLP object for one paragraph
#returns: number of normalized inflected verbs in text
def getNumInflectedVerbs(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['VBD'] + pos_freq['VBG'] + pos_freq['VBN'] + pos_freq['VBP'] + pos_freq['VBZ'])/pos_freq['SUM']

#input: NLP object for one paragraph
#returns: number of normalized determiners in text
def getNumDeterminers(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['DT'] + pos_freq['PDT'] + pos_freq['WDT'] )/pos_freq['SUM']


#input: NLP object for one paragraph
#returns: number of normalized adverbs in text
def getNumAdverbs(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['RB'] + pos_freq['RBR'] + pos_freq['RBS'] + pos_freq['WRB'] )/pos_freq['SUM']


#input: NLP object for one paragraph
#returns: number of normalized adjectives in text
def getNumAdjectives(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['JJ'] + pos_freq['JJR'] + pos_freq['JJS'])/pos_freq['SUM']

#input: NLP object for one paragraph
#returns: number of normalized interjections in text
def getNumInterjections(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['UH'])/pos_freq['SUM']  



#input: NLP object for one paragraph
#returns: number of normalized subordinate conjunctions in text
def getNumSubordinateConjunctions(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['IN'])/pos_freq['SUM']  



#input: NLP object for one paragraph
#returns: number of normalized coordinate conjunctions in text
def getNumCoordinateConjunctions(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	return  (pos_freq['CC'])/pos_freq['SUM']  



"""
===========================================================

WORD TYPE RATIOS

===========================================================
"""



#input: NLP object for one paragraph
#returns: ratio of nouns to verbs in the paragraph
def getRatioVerb(nlp_obj):

	pos_freq = nlp_obj['pos_freq']

	return  (pos_freq['NN'] + pos_freq['NNP'] + pos_freq['NNS']+ pos_freq['NNPS'])/(pos_freq['VB'] + pos_freq['VBD'] + pos_freq['VBG'] + pos_freq['VBN'] + pos_freq['VBP'] + pos_freq['VBZ'])


#input: NLP object for one paragraph
#returns: ratio of nouns to verbs in the paragraph
def getRatioNoun(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	num_nouns = pos_freq['NN'] + pos_freq['NNP'] + pos_freq['NNS']+ pos_freq['NNPS']
	num_verbs = pos_freq['VB'] + pos_freq['VBD'] + pos_freq['VBG'] + pos_freq['VBN'] + pos_freq['VBP'] + pos_freq['VBZ']

	return  num_nouns/(num_nouns + num_verbs)



#input: NLP object for one paragraph
#returns: ratio of pronouns to nouns in the paragraph
def getRatioPronoun(nlp_obj):

	pos_freq = nlp_obj['pos_freq']
	num_nouns = pos_freq['NN'] + pos_freq['NNP'] + pos_freq['NNS']+ pos_freq['NNPS']
	num_pronouns = pos_freq['PRP'] + pos_freq['PRP$'] + pos_freq['PRP'] + pos_freq['WHP'] + pos_freq['WP$']

	return  num_nouns/(num_nouns + num_verbs)


#input: NLP object for one paragraph
#returns: ratio of coordinate- to subordinate conjunctions in the paragraph
def getRatioPronoun(nlp_obj):


	return  pos_freq['CC']/pos_freq['IN']


#input: NLP object for one paragraph
#returns: ratio of  types to tokens
def getTTR(nlp_obj):

	num_types = len(set(nlp_obj(['tokens'])))
	num_words = len(nlp_obj['tokens'])

	return num_types/num_words


#input: NLP object for one paragraph
#returns: average ratio of types to tokens using a sliding window
def getMATTR(nlp_obj):

	window = 20
	total_len = len(nlp_obj['tokens'])

	words_table = Counter(nlp_obj['tokens'][0:window])
	uniq = len(set(words_table))

	moving_ttr = list([uniq/window])

	for i in range(window,total_len) 

		word_to_remove = nlp_obj['tokens'][i-window]
		words_table[word_to_remove] -= 1
		
		words_table[word_to_remove] is 0:

			uniq -= 1

		next_word =  nlp_obj['tokens'][i]
		words_table[next_word] += 1

		words_table[next_word] is 1:

			uniq += 1

		moving_ttr.append(uniq/window)


	return sum(moving_ttr)/len(moving_ttr)





#input: NLP object for one paragrah
#returns: Brunet index for that paragraph
def getBrunetIndex(nlp_obj):

	#number of word types
	word_types = len(set(nlp_obj['tokens']))

	#number of words
	words = len(nlp_obj['tokens'])

	#Brunet's index
	return words^(word_types*-0.165)

#input: NLP object for one paragrah
#returns: Honore statistic for that paragraph
def getHonoreStatistic(nlp_obj):

	#number of word types
	word_types = len(set(nlp_obj['tokens']))

	#number of words
	words = len(nlp_obj['tokens'])

	words_table = Counter(nlp_obj['tokens'])

	words_occuring_once = [word for word in nlp_obj['tokens'] if words_table[word] == 1]

	return (100*math.log(words))/(1-words_occuring_once/word_types)

#input: NLP object for one paragrah
#returns: Mean word length
def getMeanWordLength(nlp_obj):


	tokens = nlp_obj['tokens']

	word_length = [len(word) for word in tokens]

	return sum(word_length)/len(tokens)

#input: NLP object for one paragrah
#returns: number of NID words (length > 2) in paragraph
def getNumberOfNID(nlp_obj):

	pos_tag = nlp_obj['pos_tag']

	foreign_words = [word_pos for word_pos if len(word_pos[0]) > 2 and word_pos[1] == 'FW' ]

	return len(foreign_words)

#input: NLP object for one paragraph
#returns: normalized number of "uh" and "um"
def getDisfluencyFrequency(nlp_obj):

	tokens = nlp_obj['tokens']

	um_uh_words = [word for word in tokens if word == 'um' or word == 'uh']

	return len(um_uh_words)/len(tokens)


#input: NLP object for one paragraph
#returns: Get total number of words excluding NID and filled pauses
def getTotalNumberOfWords(nlp_obj):

	tokens = nlp_obj['tokens']
	pos_tag = nlp_obj['pos_tag']

	foreign_words = [word_pos for word_pos if word_pos[1] == 'FW' ]
	um_uh_words = [word for word in tokens if word == 'um' or word == 'uh']

	return len(tokens) - len(foreign_words) - len(um_uh_words)

#input: NLP object for one paragraph
#returns: Returns mean length of sentence w.r.t. number of words
def getMeanLengthOfSentence(nlp_obj):

	raw_text = nlp_obj['raw']
	tokens = nlp_obj['tokens']
	n_sentences = len(nltk.tokenize.sent_tokenize(raw_text))
	n_words = len(tokens)

	return n_sentences/n_words


