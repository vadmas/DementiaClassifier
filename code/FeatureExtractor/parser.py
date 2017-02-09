import os
import sys
import string 
import re
import requests
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import util
import nltk.data


# takes a long string and cleans it up and converts it into a vector to be extracted
# NOTE: Significant preprocessing was done by sed - make sure to run this script on preprocessed text

# Data structure
# data = {
#		"id1":[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]}, <--single utterance
#		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
#		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
#		 	],													  <--List of all utterances made during interview
#		"id2":[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
#		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
#		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
#		 	],
#		...
# }

# constants
PARSER_MAX_LENGTH = 50
DISFLUENCIES = ["uh", "um", "er", "ah"]

# globals
lmtzr = WordNetLemmatizer()
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
printable = set(string.printable)


def get_stanford_parse(sentence, port=9000):
    #raw = sentence['raw']
    # We want to iterate through k lines of the file and segment those lines as a session
    #pattern = '[a-zA-Z]*=\\s'
    #re.sub(pattern, '', raw)
    re.sub(r'[^\x00-\x7f]', r'', sentence)
    sentence = util.remove_control_chars(sentence)
    try:
        r = requests.post('http://localhost:' + str(port) +
                          '/?properties={\"annotators\":\"parse\",\"outputFormat\":\"json\"}', data=sentence)
    except requests.exceptions.ConnectionError, e:
        print "We received the following error in parser.get_stanford_parse():"
        print e
        print "------------------"
        print 'Did you start the Stanford server? If not, try:\n java -Xmx4g -cp "stanford/stanford-corenlp-full-2015-12-09/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000'
        print "------------------"
        sys.exit(1)
    json_obj = r.json()
    return json_obj['sentences'][0]

def _isValid(inputString):
    # Line should not contain numbers
    if(any(char.isdigit() for char in inputString)):
        return False
    # Line should not be empty
    elif not inputString.strip():
        return False
    # Line should contain characters (not only consist of punctuation)
    elif not bool(re.search('[a-zA-Z]', inputString)):
        return False
    else:
        return True


def _sentences(data):
    # Filter non-ascii
    data = filter(lambda x: x in printable, data)
    sentences = sent_detector.tokenize(data.strip())
    return sentences


def _remove_disfluencies(uttr):
    tokens = nltk.word_tokenize(uttr)
    tmp = [t for t in tokens if t.lower() not in DISFLUENCIES]
    clean = " ".join(tmp)
    if _isValid(clean):
        return clean
    else:
        return ""


# Lemmatize 
def _lemmetize(uttr):

    tokens = nltk.word_tokenize(uttr)
    tagged_words = nltk.pos_tag(tokens)
    
    lemmed_words = []
    for word, wordtype in tagged_words:
        wt = util.penn_to_wn(wordtype)
        lem = lmtzr.lemmatize(word,wt)
        lemmed_words.append(lem)

    return " ".join(lemmed_words)


# Clean uttr / Remove non ascii
def _clean_uttr(uttr):
    uttr = uttr.decode('utf-8').strip()
    uttr = re.sub(r'[^\x00-\x7f]', r'', uttr)
    return uttr


def _processUtterance(uttr):
    uttr = _clean_uttr(uttr)             # clean
    tokens = nltk.word_tokenize(uttr)    # Tokenize
    tagged_words = nltk.pos_tag(tokens)  # Tag

    # Get the frequency of every type
    pos_freq = defaultdict(int)
    for word, wordtype in tagged_words:
        pos_freq[wordtype] += 1    

    pos_freq['SUM'] = len(tokens)
    pt_list = []
    bd_list = []
    for u in util.split_string_by_words(uttr, PARSER_MAX_LENGTH):
        if u is not "":
            stan_parse = get_stanford_parse(u)
            pt_list.append(stan_parse["parse"])
            bd_list.append(stan_parse["basic-dependencies"])
    datum = {"pos": tagged_words, "raw": uttr, "token": tokens,
             "pos_freq": pos_freq, "parse_tree": pt_list, "basic_dependencies": bd_list}
    return datum


# Extract data from optima/dbank directory
def parse(filepaths):
    if type(filepaths) is str:
        filepaths = [filepaths]
    parsed_data = {}
    for filepath in filepaths:
        if os.path.exists(filepath):
            for filename in os.listdir(filepath):
                if filename.endswith(".txt"):
                    with open(os.path.join(filepath, filename)) as file:
                        print "Parsing: " + filename
                        session_utterances = []
                        for line in file:
                            uttr = _clean_uttr(line)
                            # uttr = _remove_disfluencies(uttr)
                            if _isValid(uttr):
                                session_utterances.append(_processUtterance(uttr))
                        parsed_data[filename] = session_utterances  # Add session
        else:
            print "Filepath not found: " + filepath
            print "Data may be empty"
    return parsed_data


if __name__ == '__main__':
    test = parse(['../data/test/dementia','../data/test/control'])
    print test 