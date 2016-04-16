from random import shuffle
import Driver as dvr

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
    "RatioPronoun": "Pronoun:noun ratio",
    "NP->PRP": "NP->PRP",
    # "": "Frequency",
    "NumAdverbs": "Adverbs",
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
    "proportion_below_threshold_0.5": "Cosine cutoff: 0.5",
    "NumVerbs": "Verb Frequency",
    "NumNouns": "Nouns",
    "MeanWordLength": "Word Length",
    "HonoreStatistic": "Honore's statistic",
    "NumInflectedVerbs": "Inflected verbs",
    "avg_cos_dist": "Average cosine distance",
    # "": "Verbs",
    "VPTypeRate": "VP rate",
    "keywordIUObjectWindow": "Key word: window",
    "binaryIUObjectWindow": "Info unit: window",
    "keywordIUObjectSink":"KEY WORD: sink",
    "keywordIUObjectCookie": "KEY WORD: cookie",
    "PProportion": "PP proportion",
    "PPTypeRate": "PP rate",
    "keywordIUObjectCurtains": "Key word: curtain",
    "binaryIUObjectCurtains": "Info unit: curtain",
    "binaryIUObjectCookie" : "Info unit: cookie",
    "binaryIUSubjectSink": "Info unit: sink",
    "binaryIUSubjectGirl": "Info unit: girl",
    "binaryIUObjectDishes": "Info unit: dish",
    "keywordIUObjectStool": "Key word: stool",
    "keywordIUSubjectWoman": "Key word: mother",
    "binaryIUObjectStool": "Info unit: stool",
    "binaryIUSubjectWoman": "Info unit: woman"
}


def make_arff_file(file_name, samples):
    arff_file_name = ARFF_DIR + file_name + ".arff"

    arff_file = open(arff_file_name, 'w+')
    # Write the headers
    # Write the relation
    arff_file.write('@RELATION \"' + file_name + '\"\n\n')
    # Assuming that all samples will have the same features
    # Assuming that all sample features are iterated in the same order
    shuffle(samples) # Randomize samples
    data_unzipped = zip(*samples)
    samples = list(data_unzipped[0])
    labels = list(data_unzipped[1])
    label_order = []
    for k,v in samples[0].iteritems():
        attribute_str = '@ATTRIBUTE '
        attribute_str += str(k).strip() + ' ' + get_attribute_from_variable(v)
        label_order.append(k)
        arff_file.write(attribute_str + '\n')
    arff_file.write('@ATTRIBUTE class {Control, Dementia} \n')
    # Begin writing the data
    arff_file.write('@DATA\n')
    for sample in range(0, len(samples)):
        data_str = ''
        for k in label_order:
            data_str += str(samples[sample][k]).strip() + ','
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


# ----------------- TRAINING STUFFS ----------------- #

def train_clinical_test_clinical(clinical_samples):
    # Use all the features
    file_name = "clinical_clinical_all"
    make_arff_file(file_name, clinical_samples)


def train_all_test_all(all_samples):
    # USE ALL THE FEATURES
    file_name = "all_all_all"
    make_arff_file(file_name, all_samples)


def train_clinical_test_clinical_fraser_features(clinical_samples):
    # Use features found in fraser
    file_name = "clinical_clinical_fraser"
    # Comb the features in clinical_samples to match the ones in fraser
    unzipped_data = zip(*clinical_samples)
    samples = list(unzipped_data[0])
    labels = list(unzipped_data[1])
    fraser_samples = []
    for sample in samples:
        features = {}
        for k,v in sample.iteritems():
            if k in fraser_feature_dict.keys():
                features[fraser_feature_dict[k]] = v
        fraser_samples.append(features)
    samples = zip(fraser_samples, labels)
    make_arff_file(file_name, samples)



#def train_all_test_all_fraser(all_samples):




if __name__ == "__main__":

    # Load the dementia and optima bank data sets.
    clinical_samples = dvr.get_clinical_feature_data()
    train_clinical_test_clinical(clinical_samples)

    # Load the clinical and non-clinical data sets
    all_samples = dvr.get_all_feature_data()
    train_all_test_all(all_samples)

    # Load clinical and test on clinical with fraser features
    train_clinical_test_clinical_fraser_features(clinical_samples)

