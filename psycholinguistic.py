#------------------------------------------------
# For testing
#------------------------------------------------

# if __name__ == '__main__':
# 	s0 = "this little boy here is taking cookies "
# 	s1 = " This is a second sentence "
# 	s2 = "This. Sentence has punctuation!"
# 	s3 = "And this sentsce has spelling mistkaes"
# 	s4 = "this little boy here is also taking cookies "
# 	s5  = "An elephant fish pork monkey"
# 	l = [s0, s1, s2, s3, s4, s5]
# 	print 'avg_cos_dist', avg_cos_dist(l)
# 	print 'min_cos_dist', min_cos_dist(l)
# 	print 'proportion_below_threshold', proportion_below_threshold(l,0)
# 	print 'proportion_below_threshold', proportion_below_threshold(l,0.3)
# 	print 'proportion_below_threshold', proportion_below_threshold(l,0.5)

import nltk
import requests
import re
import csv
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string

import parser
import extractor

# Global psycholinguistic data structures
FEATURE_DATA_PATH = 'data/feature_data/' 

# Made global so files only need to be read once
psycholinguistic_scores = {}
SUBTL_cached_scores = {}

#-----------Global Tools--------------
#remove punctuation, lowercase, stem
stemmer = nltk.PorterStemmer()
stop = stopwords.words('english')
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]
#-------------------------------------

# Constants 
LIGHT_VERBS       = ["be", "have", "come", "go", "give", "take", "make", "do", "get", "move", "put", ]
VERB_POS_TAGS     = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
NOUN_POS_TAGS     = ["NN", "NNS", "NNP", "NNPS",]
FEATURE_DATA_LIST = ["familiarity", "concreteness", "imagability",'aoa']

#Information Unit Words
BOY       =  ['boy','son','brother','male child']
GIRL      =  ['girl','daughter','sister', 'female child']
WOMAN     =  ['woman','mom','mother','lady','parent']
KITCHEN   =  ['kitchen']
EXTERIOR  =  ['exterior', 'outside', 'garden', 'yard']
COOKIE    =  ['cookie']
JAR       =  ['jar']
STOOL     =  ['stool']
SINK      =  ['sink']
PLATE     =  ['plate']
DISHCLOTH =  ['dishcloth','rag','cloth','napkin','towel']
WATER     =  ['water']
WINDOW    =  ['window']
CUPBOARD  =  ['cupboard']
DISHES    =  ['dishes']
CURTAINS  =  ['curtains','curtain']

#================================================
#-------------------Tools------------------------
#================================================
def getAllWordsFromInterview(interview):
	words = []
	for uttr in interview: 
		words += uttr["token"]
	return words

def getAllNonStopWordsFromInterview(interview):
	words = []
	for uttr in interview: 
		words += [w for w in uttr["token"] if w not in stop]
	return words

#================================================
#-----------Psycholinguistic features------------
#================================================

# Input: one of "familiarity", "concreteness", "imagability", or 'aoa'
# Output: none
# Notes: Makes dict mapping words to score, store dict in psycholinguistic
def _load_scores(name):
	if name not in FEATURE_DATA_LIST:
		raise ValueError("name must be one of: " + str(FEATURE_DATA_LIST))
	with open(FEATURE_DATA_PATH+name) as file:
		d = {word.lower():float(score) for (score, word) in [line.strip().split(" ") for line in file]}
		psycholinguistic_scores[name] = d

# Input: Interview is a list of utterance dictionaries, measure is one of "familiarity", "concreteness", "imagability", or 'aoa'
# Output: PsycholinguisticScore for a given measure 
def getPsycholinguisticScore(interview, measure):
	if measure not in FEATURE_DATA_LIST:
		raise ValueError("name must be one of: " + str(FEATURE_DATA_LIST))
	if measure not in psycholinguistic_scores: 
		_load_scores(measure)
	score = 0
	allwords = getAllNonStopWordsFromInterview(interview)
	for w in allwords:
		if w.lower() in psycholinguistic_scores[measure]:
			score += psycholinguistic_scores[measure][w.lower()]
	return score / len(allwords)
	 
# Input: list of words
# Output: scores for each word
# Notes: This gets the SUBTL frequency count for a word from http://subtlexus.lexique.org/moteur2/index.php
def _getSUBTLWordScoresFromURL(wordlist):
	unknown_words = [w for w in wordlist if w not in SUBTL_cached_scores]
	# Load into cache all unknown words
	if unknown_words:
		url = 'http://subtlexus.lexique.org/moteur2/simple.php'
		encoded_words = '\n'.join(unknown_words)
		params = {'database':'subtlexus', 'mots':encoded_words}
		r = requests.get(url, params=params)
		rows = []
		table = BeautifulSoup(r.content,"html.parser")
		# Parse datatable to get SUBTLwf scores
		for row in table.findAll("tr"):
			cells = row.findAll("td")
			row = [c.findAll(text=True)[0] for c in cells[:10]]
			rows.append(row)
		# Fill dictionary, ignore header row 
		for row in rows[1:]:
			SUBTL_cached_scores[row[0]] = float(row[5])

	# Read the scores for each word
	# (Ignores words which don't have score)
	return [SUBTL_cached_scores[w] for w in wordlist if w in SUBTL_cached_scores]

def getSUBTLWordScores(interview):
	allwords = getAllNonStopWordsFromInterview(interview)
	scores = _getSUBTLWordScoresFromURL(allwords)
	return sum(scores) / len(allwords)

# Input: Sent is a sentence dictionary,
# Output: Normalized count of light verbs
def getLightVerbCount(interview):
	light_verbs = 0.0
	total_verbs = 0.0
	for uttr in interview:
		for w in uttr['pos']:
			if w[0].lower in LIGHT_VERBS: 
				light_verbs += 1 
			if w[1] in VERB_POS_TAGS:
				total_verbs += 1
	return light_verbs/total_verbs

#================================================
#-----------Information Unit features------------
# Taken from http://www.sciencedirect.com/science/article/pii/S0093934X96900334
# End of page 4
#================================================
# Input: Sent is a sentence dictionary,
# Output: Binary value (1/0) of sentence contains info unit
# Notes: Info units are hard coded keywords (Need to find paper or something to justify hard coded words)


#--------------
# Subjects (3)
#-------------

def keywordIUSubjectBoy(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in BOY]
	return len(keywords)

def binaryIUSubjectBoy(interview):
	count = keywordIUSubjectBoy(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUSubjectGirl(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in GIRL]
	return len(keywords)

def binaryIUSubjectGirl(interview):
	count = keywordIUSubjectGirl(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUSubjectWoman(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in WOMAN]
	return len(keywords)

def binaryIUSubjectWoman(interview):
	count = keywordIUSubjectWoman(interview)
	return 0 if count == 0 else 1


#--------------
# Places (2)
#-------------


def keywordIUPlaceKitchen(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in KITCHEN]
	return len(keywords)

def binaryIUPlaceKitchen(interview):
	count = keywordIUPlaceKitchen(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUPlaceExterior(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in EXTERIOR]
	return len(keywords)

def binaryIUPlaceExterior(interview):
	count = keywordIUPlaceExterior(interview)
	return 0 if count == 0 else 1


#--------------
# Objects (11)
#-------------


def keywordIUObjectCookie(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in COOKIE]
	return len(keywords)

def binaryIUObjectCookie(interview):
	count = keywordIUObjectCookie(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectJar(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in JAR]
	return len(keywords)

def binaryIUObjectJar(interview):
	count = keywordIUObjectJar(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectStool(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in STOOL]
	return len(keywords)

def binaryIUObjectStool(interview):
	count = keywordIUObjectStool(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectSink(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in SINK]
	return len(keywords)

def binaryIUObjectSink(interview):
	count = keywordIUObjectSink(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectPlate(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in PLATE]
	return len(keywords)

def binaryIUObjectPlate(interview):
	count = keywordIUObjectPlate(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectDishcloth(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in DISHCLOTH]
	return len(keywords)

def binaryIUObjectDishcloth(interview):
	count = keywordIUObjectDishcloth(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectWater(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in WATER]
	return len(keywords)

def binaryIUObjectWater(interview):
	count = keywordIUObjectWater(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectWindow(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in WINDOW]
	return len(keywords)

def binaryIUObjectWindow(interview):
	count = keywordIUObjectWindow(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectCupboard(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in CUPBOARD]
	return len(keywords)

def binaryIUObjectCupboard(interview):
	count = keywordIUObjectCupboard(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectDishes(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in DISHES]
	return len(keywords)

def binaryIUObjectDishes(interview):
	count = keywordIUObjectDishes(interview)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectCurtains(interview):
	words = getAllWordsFromInterview(interview)
	keywords = [w for w in words if w in CURTAINS]
	return len(keywords)

def binaryIUObjectCurtains(interview):
	count = keywordIUObjectCurtains(interview)
	return 0 if count == 0 else 1


#--------------
# Actions (7)
#-------------

# For action unit to be present, the subject and action (eg. 'boy' and 'fall') 
# must be tagged together in the utterance
# Input: POSTags, subject list, verb list
def check_action_unit(pos_tags, subjs, verbs):
	stemmed_subjs = [stemmer.stem(s) for s in subjs]
	stemmed_verbs = [stemmer.stem(s) for s in verbs]
	subj_found, verb_found = False, False
	for pos in pos_tags:
		if stemmer.stem(pos[0]) in stemmed_subjs and pos[1] in NOUN_POS_TAGS: subj_found = True
		if stemmer.stem(pos[0]) in stemmed_verbs and pos[1] in VERB_POS_TAGS: verb_found = True
	return subj_found and verb_found


#boy taking or stealing
def binaryIUActionBoyTaking(interview):
	for uttr in interview:
		if(check_action_unit(uttr['pos'],BOY,['take', 'steal'])):
			return True
	return False

# boy or stool falling
def binaryIUActionStoolFalling(interview):
	for uttr in interview:
		if(check_action_unit(uttr['pos'],BOY+['stool'],['falling'])):
			return True
	return False

# Woman drying or washing dishes/plate
def binaryIUActionWomanDryingWashing(interview):
	for uttr in interview:
		if(check_action_unit(uttr['pos'],WOMAN+['dish','plate'], ['wash','dry'])):
			return True
	return False

# Water overflowing or spilling
def binaryIUActionWaterOverflowing(interview):
	for uttr in interview:
		if(check_action_unit(uttr['pos'],['water','tap','sink'],['overflow', 'spill'])):
			return True
	return False

#??????????????????????????????
#How to define 'action?'
#??????????????????????????????

# #action performed by the girl,
# def binaryIUActionGirl(interview):
# 	return check_action_unit(Tree.interview(tree),GIRL,['asking','reaching','helping'])

# #woman unconcerned by the overflowing,
# def binaryIUActionWomanUnconcerned(interview):
# 	return check_action_unit(Tree.fromstring(interview),WOMAN,['unconcerned'])

# #woman indifferent to the children. 
# def binaryIUActionWomanIndifferent(interview):
# 	return check_action_unit(Tree.fromstring(interview),['stool'],['falling'])


#-------------------------------------
# Cosine Similarity Between Utterances
#-------------------------------------
def not_only_stopwords(text):
	unstopped = [w for w in normalize(text) if w not in stop]
	return len(unstopped) != 0 

def normalize(text):
	remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
	return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

#input: two strings 
#returns: (float) similarity 
#Note: returns zero if one string consists only of stopwords
def cosine_sim(text1, text2):
	if not_only_stopwords(text1) and not_only_stopwords(text2):
		vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')	# Punctuation remover
		tfidf = vectorizer.fit_transform([text1, text2])
		return ((tfidf * tfidf.T).A)[0,1]
	else:
		return 0
#input: list of raw utterances
#returns: list of cosine similarity between all pairs
def compare_all_utterances(uttrs):
	similarities = []
	for i in range(len(uttrs)):
		for j in range(i+1,len(uttrs)):
			similarities.append(cosine_sim(uttrs[i]['raw'],uttrs[j]['raw']))
	return similarities

#input: list of raw utterances
#returns: (float)average similarity over all similarities
def avg_cos_dist(uttrs):
	similarities = compare_all_utterances(uttrs)
	return reduce(lambda x,y: x+y, similarities)/len(similarities)

#input: list of raw utterances
#returns:(float) Minimum similarity over all similarities
def min_cos_dist(uttrs):
	return min(compare_all_utterances(uttrs))

#input: list of raw utterances
#returns: (float) proportion of similarities below threshold
def proportion_below_threshold(uttrs,thresh):
	similarities = compare_all_utterances(uttrs)
	valid = [s for s in similarities if s <= thresh]
	return len(valid)/float(len(similarities))

#input: list of interview utterances stored as [ [{},{},{}], [{},{},{}] ]
#returns: list of features for each interview
def get_all_features(data):
	feature_set = []
	for idx, datum in enumerate(data):
		print "Extracting psycholinguistic features for:", idx
		features = []
		features.append(getPsycholinguisticScore(datum,'familiarity'))
		features.append(getPsycholinguisticScore(datum,'concreteness'))
		features.append(getPsycholinguisticScore(datum,'imagability'))
		features.append(getPsycholinguisticScore(datum,'aoa'))
		features.append(getSUBTLWordScores(datum))
		features.append(getLightVerbCount(datum))
		features.append(keywordIUSubjectBoy(datum))		  
		#Boy IU
		features.append(binaryIUSubjectBoy(datum))
		features.append(keywordIUSubjectGirl(datum))	  
		#Girl IU
		features.append(binaryIUSubjectGirl(datum))
		features.append(keywordIUSubjectWoman(datum))     
		#Woman IU
		features.append(binaryIUSubjectWoman(datum))
		features.append(keywordIUPlaceKitchen(datum))	  
		#Kitchen IU
		features.append(binaryIUPlaceKitchen(datum))
		features.append(keywordIUPlaceExterior(datum))	  
		#Exterior IU
		features.append(binaryIUPlaceExterior(datum))
		features.append(keywordIUObjectCookie(datum))	  
		#Cookie IU
		features.append(binaryIUObjectCookie(datum))
		features.append(keywordIUObjectJar(datum))		  
		#Jar IU
		features.append(binaryIUObjectJar(datum))
		features.append(keywordIUObjectStool(datum))	  
		#Stool IU
		features.append(binaryIUObjectStool(datum))
		features.append(keywordIUObjectSink(datum))		  
		#Sink IU
		features.append(binaryIUObjectSink(datum))
		features.append(keywordIUObjectPlate(datum))	  
		#Plate IU
		features.append(binaryIUObjectPlate(datum))
		features.append(keywordIUObjectDishcloth(datum))  
		#Dishcloth IU
		features.append(binaryIUObjectDishcloth(datum))
		features.append(keywordIUObjectWater(datum))	  
		#Water IU
		features.append(binaryIUObjectWater(datum))
		features.append(keywordIUObjectWindow(datum))	  
		#Window IU
		features.append(binaryIUObjectWindow(datum))
		features.append(keywordIUObjectCupboard(datum))	  
		#Cupboard IU
		features.append(binaryIUObjectCupboard(datum))
		features.append(keywordIUObjectDishes(datum))	  
		#Dishes IU
		features.append(binaryIUObjectDishes(datum))
		features.append(keywordIUObjectCurtains(datum))   
		#Curtains IU
		features.append(binaryIUObjectCurtains(datum))
		features.append(binaryIUActionBoyTaking(datum))	  
		#Boy taking IU
		features.append(binaryIUActionStoolFalling(datum))
		#Stool falling taking IU
		features.append(binaryIUActionWomanDryingWashing(datum))
		features.append(binaryIUActionWaterOverflowing(datum))
		features.append(binaryIUActionWaterOverflowing(datum))
		features.append(binaryIUActionWaterOverflowing(datum))
		features.append(binaryIUActionWaterOverflowing(datum))
		features.append(avg_cos_dist(datum))
		features.append(min_cos_dist(datum))
		features.append(proportion_below_threshold(datum,0))
		features.append(proportion_below_threshold(datum,0.3))
		features.append(proportion_below_threshold(datum,0.5))
		
		# Append feature vector to set
		feature_set.append(features)
	return feature_set

#------------------------------------------------
# For testing
#------------------------------------------------

# if __name__ == '__main__':
# 	s0 = "this little boy here is taking cookies "
# 	s1 = " This is a second sentence "
# 	s2 = "This. Sentence has punctuation!"
# 	s3 = "And this sentsce has spelling mistkaes"
# 	s4 = "this little boy here is also taking cookies "
# 	s5  = "An elephant fish pork monkey"
# 	l = [s0, s1, s2, s3, s4, s5]
# 	print 'avg_cos_dist', avg_cos_dist(l)
# 	print 'min_cos_dist', min_cos_dist(l)
# 	print 'proportion_below_threshold', proportion_below_threshold(l,0)
# 	print 'proportion_below_threshold', proportion_below_threshold(l,0.3)
# 	print 'proportion_below_threshold', proportion_below_threshold(l,0.5)

# 	# # print avg_cos_dist(l)