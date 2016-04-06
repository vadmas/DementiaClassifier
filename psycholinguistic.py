import nltk
import requests
import re
import csv
import requests
from bs4 import BeautifulSoup
from nltk.tree import *
from sklearn.feature_extraction.text import TfidfVectorizer
import string

import parser
import extractor

# Global psycholinguistic data structures
FEATURE_DATA_PATH = 'data/feature_data/' 

# Made global so files only need to be read once
psycholinguistic_scores = {}

#-----------Global Tools--------------
#remove punctuation, lowercase, stem
stemmer = nltk.PorterStemmer()
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
#--------------Parse Tree Methods---------------
#================================================

# For action unit to be present, the subject and action (eg. 'boy' and 'fall') 
# must be tagged together in the utterance
# Input: NLTK.Tree, subject list, verb list
def check_action_unit(tree, subjs, verbs):
	stemmed_subjs = [stemmer.stem(s) for s in subjs]
	stemmed_verbs = [stemmer.stem(s) for s in verbs]
	subj_found, verb_found = False, False
	for pos in tree.pos():
		if stemmer.stem(pos[0]) in stemmed_subjs and pos[1] in NOUN_POS_TAGS: subj_found = True
		if stemmer.stem(pos[0]) in stemmed_verbs and pos[1] in VERB_POS_TAGS: verb_found = True
	return subj_found and verb_found

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
		d = {word.lower():score for (score, word) in [line.strip().split(" ") for line in file]}
		psycholinguistic[name] = d

# Input: Sent is a sentence dictionary, measure is one of "familiarity", "concreteness", "imagability", or 'aoa'
# Output: PsycholinguisticScore for a given measure 
def getPsycholinguisticScore(sent, measure):
	if measure not in FEATURE_DATA_LIST:
		raise ValueError("name must be one of: " + str(FEATURE_DATA_LIST))
	if not psycholinguistic[measure]: 
		_load_scores(measure)
	score = 0
	for w in sent['token']:
		if w.lower() in psycholinguistic[measure]:
			score += psycholinguistic[measure][w.lower]
	return score/len(sent['token'])
	 
# Input: list of words
# Output: scores for each word
# Notes: This gets the SUBTL frequency count for a word from 
# 'SubtlexUS: American Word Frequencies'
# http://subtlexus.lexique.org/moteur2/index.php
# Improvements: Should add caching here to remove repeated calls 
def getSUBTLWordScores(wordlist):
	url = 'http://subtlexus.lexique.org/moteur2/simple.php'
	encoded_words = '\n'.join(wordlist)
	params = {'database':'subtlexus', 'mots':encoded_words}
	r = requests.get(url, params=params)
	vals = []
	table = BeautifulSoup(r.content,"html.parser")
	for row in table.findAll("tr"):
		cells = row.findAll("td")
		row = [c.findAll(text=True) for c in cells[:10]]
		vals.append(row)
	return vals

# Input: Sent is a sentence dictionary,
# Output: Normalized count of light verbs
def getLightVerbCount(sent):
	light_verbs = 0
	total_verbs = 0
	for w in sent['pos']:
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


def keywordIUSubjectBoy(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in BOY:
			count +=1
	return count

def binaryIUSubjectBoy(sent):
	count = keywordIUSubjectBoy(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUSubjectGirl(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in GIRL:
			count +=1
	return count

def binaryIUSubjectGirl(sent):
	count = keywordIUSubjectGirl(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUSubjectWoman(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in WOMAN:
			count +=1
	return count

def binaryIUSubjectWoman(sent):
	count = keywordIUSubjectWoman(sent)
	return 0 if count == 0 else 1


#--------------
# Places (2)
#-------------


def keywordIUPlaceKitchen(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in KITCHEN:
			count +=1
	return count

def binaryIUPlaceKitchen(sent):
	count = keywordIUPlaceKitchen(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUPlaceExterior(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in EXTERIOR:
			count +=1
	return count

def binaryIUPlaceExterior(sent):
	count = keywordIUPlaceExterior(sent)
	return 0 if count == 0 else 1


#--------------
# Objects (11)
#-------------


def keywordIUObjectCookie(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in COOKIE:
			count +=1
	return count

def binaryIUObjectCookie(sent):
	count = keywordIUObjectCookie(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectJar(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in JAR:
			count +=1
	return count

def binaryIUObjectJar(sent):
	count = keywordIUObjectJar(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectStool(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in STOOL:
			count +=1
	return count

def binaryIUObjectStool(sent):
	count = keywordIUObjectStool(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectSink(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in SINK:
			count +=1
	return count

def binaryIUObjectSink(sent):
	count = keywordIUObjectSink(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectPlate(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in PLATE:
			count +=1
	return count

def binaryIUObjectPlate(sent):
	count = keywordIUObjectPlate(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectDishcloth(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in DISHCLOTH:
			count +=1
	return count

def binaryIUObjectDishcloth(sent):
	count = keywordIUObjectDishcloth(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectWater(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in WATER:
			count +=1
	return count

def binaryIUObjectWater(sent):
	count = keywordIUObjectWater(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectWindow(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in WINDOW:
			count +=1
	return count

def binaryIUObjectWindow(sent):
	count = keywordIUObjectWindow(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectCupboard(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in CUPBOARD:
			count +=1
	return count

def binaryIUObjectCupboard(sent):
	count = keywordIUObjectCupboard(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectDishes(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in DISHES:
			count +=1
	return count

def binaryIUObjectDishes(sent):
	count = keywordIUObjectDishes(sent)
	return 0 if count == 0 else 1

#-----

def keywordIUObjectCurtains(sent):
	count = 0
	for w in sent['token']:
		if w.lower() in CURTAINS:
			count +=1
	return count

def binaryIUObjectCurtains(sent):
	count = keywordIUObjectCurtains(sent)
	return 0 if count == 0 else 1


#--------------
# Actions (7)
#-------------

#boy taking or stealing
def binaryIUActionBoyTaking(tree):
	return check_action_unit(Tree.fromstring(tree),BOY,['take', 'steal'])

# boy or stool falling
def binaryIUActionStoolFalling(tree):
	return check_action_unit(Tree.fromstring(tree),BOY+['stool'],['falling'])

# Woman drying or washing dishes/plate
def binaryIUActionWomanDryingWashing(tree):
	return check_action_unit(Tree.fromstring(tree), WOMAN+['dish','plate'], ['wash','dry'])

# Water overflowing or spilling
def binaryIUActionWaterOverflowing(tree):
	return check_action_unit(Tree.fromstring(tree),['water','tap','sink'],['overflow', 'spill'])

#??????????????????????????????
#How to define 'action?'
#??????????????????????????????

# #action performed by the girl,
# def binaryIUActionGirl(tree):
# 	return check_action_unit(Tree.fromstring(tree),GIRL,['asking','reaching','helping'])

# #woman unconcerned by the overflowing,
# def binaryIUActionWomanUnconcerned(tree):
# 	return check_action_unit(Tree.fromstring(tree),WOMAN,['unconcerned'])

# #woman indifferent to the children. 
# def binaryIUActionWomanIndifferent(tree):
# 	return check_action_unit(Tree.fromstring(tree),['stool'],['falling'])


#-------------------------------------
# Cosine Similarity Between Utterances
#-------------------------------------
def normalize(text):
	remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
	return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

#input: two strings 
#returns: (float) similarity 
def cosine_sim(text1, text2):
	vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')	# Punctuation remover
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

#input: list of raw utterances
#returns: list of cosine similarity between all pairs
def compare_all_utterances(uttrs):
	similarities = []
	for i in range(len(uttrs)):
		for j in range(i+1,len(uttrs)):
			similarities.append(cosine_sim(uttrs[i],uttrs[j]))
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