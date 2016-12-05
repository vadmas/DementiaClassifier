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

def print_full(x):
    for l in x:
        print l


def get_column_index(columns, df):
    return [df.columns.get_loc(c) for c in columns]


def shorten(name):
    if "keyword_to_non_keyword_ratio" in name:
        name = name.replace("keyword_to_non_keyword_ratio", 'kw_to_w_ratio')

    if "keyword_type_to_token" in name:
        name = name.replace("keyword_type_to_token", "ty_to_tok")

    if "percentage_of_leftside_keywords_mentioned" in name:
        name = name.replace("percentage_of_leftside_keywords_mentioned", "prcnt_ls_uttered")

    if "percentage_of_rightside_keywords_mentioned" in name:
        name = name.replace("percentage_of_rightside_keywords_mentioned", "prcnt_rs_uttered")

    if "count_of_leftside_keyword" in name:
        name = name.replace("count_of_leftside_keyword", "ls_count")

    if "count_of_rightside_keyword" in name:
        name = name.replace("count_of_rightside_keyword", "rs_count")

    if "percentage_of_rightside_keywords_mentioned" in name:
        name = name.replace("percentage_of_rightside_keywords_mentioned", "prcnt_rs_uttered")

    if "leftside" in name:
        name = name.replace("leftside", "ls")

    if "rightside" in name:
        name = name.replace("rightside", "rs")

    return name
