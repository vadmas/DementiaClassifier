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

feature_dict = {
    # PARSE TREE FEATURES
    # PSYCHOLINGUISTIC FEATURES
    # POS FEATURES
    "Pronoun:noun ratio": "",
    "NP->PRP": "NP->PRP",
    "Frequency": "",
    "Adverbs": "",
    "ADVP->RB": "",
    "Verb Frequency": "",
    "Nouns": "",
    "Word Length": "",
    "NP->DT_NN": "NP->DT_NN",
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
    "VP->AUX_VP": "VP->AUX_VP",
    "VP->VBG": "VP->VBG",
    "Key word: window": "",
    "Info unit: window": "",




}


def make_arff_file(file_name, samples, labels):
    arff_file_name = ARFF_DIR + file_name + ".arff"
    arff_file = open(arff_file_name, 'w+')
    # Write the headers
    # Write the relation
    arff_file.write('@RELATION \"' + file_name + '\"\n\n')
    # Assuming that all samples will have the same features
    # Assuming that all sample features are iterated in the same order
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

if __name__ == "__main__":
    test_samples = [
        {
            'feature1': 2,
            'feature2': 10,
            'feature3': 'something',
            'feature4': 'anything'
        },
        {
            'feature1': 2,
            'feature2': 10,
            'feature3': 'something',
            'feature4': 'anything'

        }, {
            'feature1': 2,
            'feature2': 10,
            'feature3': 'something',
            'feature4': 'anything'

        }
    ]
    test_labels = ['label1', 'label2', 'label1']

    make_arff_file('test_arff', test_samples, test_labels)


