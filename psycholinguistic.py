import nltk
import requests
import re
import csv
import requests
from bs4 import BeautifulSoup

# Global psycholinguistic data structures
FEATURE_DATA_PATH = 'data/feature_data/' 

# Made global so files only need to be read once
psycholinguistic_scores = {}

# Constants 
LIGHT_VERBS       = ["be", "have", "come", "go", "give", "take", "make", "do", "get", "move", "put", ]
VERB_POS_TAGS     = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
FEATURE_DATA_LIST = ["familiarity", "concreteness", "imagability",'aoa']

#Information Unit Words
BOY       =  ['boy','son','brother']
GIRL      =  ['girl','daughter','sister']
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

def binaryIUSubjectBoy(sent):
    for w in sent['token']: if w in BOY: return 1
    return 0

def keywordIUSubjectBoy(sent):
    count = 0
    for w in sent['token']: if w in BOY: count +=1
    return count



def binaryIUSubjectGirl(sent):
    for w in sent['token']: if w in GIRL: return 1
    return 0

def keywordIUSubjectGirl(sent):
    count = 0
    for w in sent['token']: if w in GIRL: count +=1
    return count



def binaryIUSubjectWoman(sent):
    for w in sent['token']: if w in WOMAN: return 1
    return 0

def keywordIUSubjectWoman(sent):
    count = 0
    for w in sent['token']: if w in WOMAN: count +=1
    return count



#--------------
# Places (2)
#-------------

def binaryIUPlaceKitchen(sent):
    for w in sent['token']: if w in KITCHEN: return 1
    return 0

def keywordIUPlaceKitchen(sent):
    count = 0
    for w in sent['token']: if w in KITCHEN: count +=1
    return count



def binaryIUPlaceExterior(sent):
    for w in sent['token']: if w in EXTERIOR: return 1
    return 0

def keywordIUPlaceExterior(sent):
    count = 0
    for w in sent['token']: if w in EXTERIOR: count +=1
    return count



#--------------
# Objects (11)
#-------------

def binaryIUObjectCookie(sent):
    for w in sent['token']: if w in COOKIE: return 1
    return 0

def keywordIUObjectCookie(sent):
    count = 0
    for w in sent['token']: if w in COOKIE: count +=1
    return count



def binaryIUObjectJar(sent):
    for w in sent['token']: if w in JAR: return 1
    return 0

def keywordIUObjectJar(sent):
    count = 0
    for w in sent['token']: if w in JAR: count +=1
    return count



def binaryIUObjectStool(sent):
    for w in sent['token']: if w in STOOL: return 1
    return 0

def keywordIUObjectStool(sent):
    count = 0
    for w in sent['token']: if w in STOOL: count +=1
    return count



def binaryIUObjectSink(sent):
    for w in sent['token']: if w in SINK: return 1
    return 0

def keywordIUObjectSink(sent):
    count = 0
    for w in sent['token']: if w in SINK: count +=1
    return count



def binaryIUObjectPlate(sent):
    for w in sent['token']: if w in PLATE: return 1
    return 0

def keywordIUObjectPlate(sent):
    count = 0
    for w in sent['token']: if w in PLATE: count +=1
    return count



def binaryIUObjectDishcloth(sent):
    for w in sent['token']: if w in DISHCLOTH: return 1
    return 0

def keywordIUObjectDishcloth(sent):
    count = 0
    for w in sent['token']: if w in DISHCLOTH: count +=1
    return count



def binaryIUObjectWater(sent):
    for w in sent['token']: if w in WATER: return 1
    return 0

def keywordIUObjectWater(sent):
    count = 0
    for w in sent['token']: if w in WATER: count +=1
    return count



def binaryIUObjectWindow(sent):
    for w in sent['token']: if w in WINDOW: return 1
    return 0

def keywordIUObjectWindow(sent):
    count = 0
    for w in sent['token']: if w in WINDOW: count +=1
    return count



def binaryIUObjectCupboard(sent):
    for w in sent['token']: if w in CUPBOARD: return 1
    return 0

def keywordIUObjectCupboard(sent):
    count = 0
    for w in sent['token']: if w in CUPBOARD: count +=1
    return count



def binaryIUObjectDishes(sent):
    for w in sent['token']: if w in DISHES: return 1
    return 0

def keywordIUObjectDishes(sent):
    count = 0
    for w in sent['token']: if w in DISHES: count +=1
    return count



def binaryIUObjectCurtains(sent):
    for w in sent['token']: if w in CURTAINS: return 1
    return 0

def keywordIUObjectCurtains(sent):
    count = 0
    for w in sent['token']: if w in CURTAINS: count +=1
    return count



#--------------
# Actions (7)
#-------------

#boy taking or stealing
def binaryIUActionBoyTaking(sent):
    pass

#boy or stool falling
def binaryIUActionBoyStoolFalling(sent):
    pass

# Woman drying or washing dishes/plate
def binaryIUActionWomanDryingWashing(sent):
    pass

# Water overflowing or spilling
def binaryIUActionWaterOverflowing(sent):
    pass

#action performed by the girl,
def binaryIUActionGirl(sent):
    pass

#woman unconcerned by the overflowing,
def binaryIUActionWomanUnconcerned(sent):
    pass

#woman indifferent to the children. 
def binaryIUActionWomanIndifferent(sent):
    pass



if __name__ == '__main__':
