from nltk.corpus import wordnet as wn
import re
# or equivalently and much more efficiently
control_chars = ''.join(map(unichr, range(0, 32) + range(127, 160)))
control_char_re = re.compile('[%s]' % re.escape(control_chars))


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    # Return noun by default
    return wn.NOUN


def split_string_by_words(sen, n):
    tokens = sen.split()
    return [" ".join(tokens[(i) * n:(i + 1) * n]) for i in range(len(tokens) / n + 1)]


def remove_control_chars(s):
    return control_char_re.sub('', s)

def likelihood_ratio_test(alt_loss, null_loss):
    
