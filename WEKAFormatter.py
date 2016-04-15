from random import shuffle

ARFF_DIR = 'arff files/'
FEATURES_DIR = '/'


attribute_dict = {
    'int': 'numeric',
    'float': 'numeric',
    'long': 'numeric',
    'str': 'string'
}


# ------------------------
# Fill these in the corresponding key names you guys use
# ------------------------

fraser_feature_dict = {
    # PARSE TREE FEATURES
    # PSYCHOLINGUISTIC FEATURES
    # POS FEATURES
    "Pronoun:noun ratio": "",
    "NP->PRP": "NP->PRP",
    "Frequency": "",
    "Adverbs": "",
    "ADVP->RB": "ADVP->RB",
    "VP->VBG_PP": "VP->VBG_PP",
    "VP->IN_S": "VP->IN_S",
    "VP->AUX_ADJP": "VP->AUX_ADJP",
    "VP->AUX_VP": "VP->AUX_VP",
    "VP->VBG": "VP->VBG",
    "VP->AUX": "VP->AUX",
    "VP->VBD_NP": "VP->VBD_NP",
    "INTJ->UH": "INTJ->UH",
    "NP->DT_NN": "NP->DT_NN",
    "Cosine cutoff: 0.5": "",
    "Verb Frequency": "",
    "Nouns": "",
    "Word Length": "",
    "Honore's statistic": "",
    "Inflected verbs": "",
    "Average cosine distance": "",
    "Skewness(MFCC 1)": "",
    "Skewness(MFCC 2)": "",
    "Kurtosis(MFCC 5)": "",
    "Kurtosis(VEL(MFCC 3))": "",
    "Phonation rate": "",
    "Skewness(MFCC 8)": "",
    "Verbs": "",
    "VP rate": "",
    "Key word: window": "",
    "Info unit: window": "",
    "KEY WORD: sink": "",
    "KEY WORD: cookie": "",
    "PP proportion": "",
    "Key word: curtain": "",
    "PP rate": "",
    "Info unit: curtain": "",
    "Key word: counter": "",
    "Info unit: cookie": "",
    "Info unit: sink" : "",
    "Info unit: girl": "",
    "Info unit: girl’s action": "",
    "Info unit: dish": "",
    "Key word: stool": "",
    "Key word: mother": "",
    "Info unit: stool": "",
    "Skewness(MFCC 12)": "",
    "Info unit: woman": "",
}


def make_arff_file(file_name, samples, labels):
    arff_file_name = ARFF_DIR + file_name + ".arff"
    arff_file = open(arff_file_name, 'w+')
    # Write the headers
    # Write the relation
    arff_file.write('@RELATION \"' + file_name + '\"\n\n')
    # Assuming that all samples will have the same features
    # Assuming that all sample features are iterated in the same order
    shuffle(samples) # Randomize samples
    for k,v in samples[0].iteritems():
        attribute_str = '@ATTRIBUTE '
        attribute_str += str(k) + ' ' + get_attribute_from_variable(v)
        arff_file.write(attribute_str + '\n')
    arff_file.write('@ATTRIBUTE class {Control, Dementia} \n')
    # Begin writing the data
    arff_file.write('@DATA\n')
    for sample in range(0, len(samples)):
        data_str = ''
        for k,v in samples[sample].iteritems():
            data_str += str(v) + ','
        data_str += labels[sample]
        arff_file.write(data_str + '\n')
    arff_file.close()


def get_attribute_from_variable(var):

    if is_number(var):
        return "numeric"
    else:
        return "str"


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def train_clinical_test_clinical(clinical_samples, labels):
    # Use all the features
    file_name = ARFF_DIR + "train_clinical_test_clinical.arff"
    make_arff_file(file_name, clinical_samples, labels)



if __name__ == "__main__":
    # Load the dementia and optima bank data sets.


