# Use cPickle if available
try:
    import cPickle as pickle
except:
    import pickle
import os
from FeatureExtractor import parser 
from FeatureExtractor import pos_phrases 
from FeatureExtractor import pos_syntactic 
from FeatureExtractor import psycholinguistic
import random

# constants
# Unsplit data
DEMENTIABANK_CONTROL_DIR_ALL  = 'data/dbank/control'
DEMENTIABANK_DEMENTIA_DIR_ALL = 'data/dbank/dementia'
OPTIMA_CONTROL_DIR_ALL        = 'data/optima/nometa/control'
OPTIMA_DEMENTIA_DIR_ALL       = 'data/optima/nometa/dementia'

# Training data
DEMENTIABANK_CONTROL_DIR_TRAIN  = 'data/dbank/split/train/control'
DEMENTIABANK_DEMENTIA_DIR_TRAIN = 'data/dbank/split/train/dementia'
OPTIMA_CONTROL_DIR_TRAIN        = 'data/optima/split/train/control'
OPTIMA_DEMENTIA_DIR_TRAIN       = 'data/optima/split/train/dementia'

# Test data
DEMENTIABANK_CONTROL_DIR_TEST  = 'data/dbank/split/test/control'
DEMENTIABANK_DEMENTIA_DIR_TEST = 'data/dbank/split/test/dementia'
OPTIMA_CONTROL_DIR_TEST        = 'data/optima/split/test/control'
OPTIMA_DEMENTIA_DIR_TEST       = 'data/optima/split/test/dementia'

#Welcome To Our World text file
WTOW_DIR    = 'data/wtow'

#It's Just a Matter Of Balance text file
IJAMOB_DIR  = 'data/ijamob'

PICKLE_DIR  = 'data/pickles/'
#PICKLE_DIR = 'stanford/processed/pickles/'
TEST_DIR    = 'data/test/'

#Output Directory
OUTPUT_DIR = "FeatureVecs/"

#Book lines / interview
BOOK_LINES_PER_INTERVIEW = 5
# -------------Pickle functions-----------
def open_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data
    
def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def overwrite_pickle(path, data):
    f = open(path, 'wb')
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
# -------------End of pickle functions-----------


# -------------Get Data functions-----------

def read_file(picklename, raw_files_directory):
    #Check pickle first, use parser if pickle doesn't exist
    if os.path.exists(PICKLE_DIR + picklename):
        print "Pickle found at: " + PICKLE_DIR + picklename
        data = open_pickle(PICKLE_DIR + picklename)
    else:
        print "Pickle not found, beginning parse."
        data = parser.parse(raw_files_directory)
        save_pickle(PICKLE_DIR + picklename,data)
    return data

def get_book_data(picklename, raw_files_directory):
    #Check pickle first, use parser if pickle doesn't exist
    if os.path.exists(PICKLE_DIR + picklename):
        print "Pickle found at: " + PICKLE_DIR + picklename
        data = open_pickle(PICKLE_DIR + picklename)
    else:
        print "Pickle not found, beginning parse."
        data = parser.parse_book(raw_files_directory, BOOK_LINES_PER_INTERVIEW)
        save_pickle(PICKLE_DIR + picklename,data)
    return data

#dataset = all, train, or test
def get_data(dataset):
    dataset = dataset.lower()
    if dataset == 'train':
        dbank_control_train  = read_file('dbank_control_train.pickle', DEMENTIABANK_CONTROL_DIR_TRAIN)
        dbank_dem_train      = read_file('dbank_dem_train.pickle',     DEMENTIABANK_DEMENTIA_DIR_TRAIN)
        optima_control_train = read_file('optima_control_train.pickle',OPTIMA_CONTROL_DIR_TRAIN)
        optima_dem_train     = read_file('optima_dem_train.pickle',    OPTIMA_DEMENTIA_DIR_TRAIN)
        return dbank_control_train, dbank_dem_train, optima_control_train, optima_dem_train
    elif dataset == 'test':
        dbank_control_test  = read_file('dbank_control_test.pickle', DEMENTIABANK_CONTROL_DIR_TEST)
        dbank_dem_test      = read_file('dbank_dem_test.pickle',     DEMENTIABANK_DEMENTIA_DIR_TEST)
        optima_control_test = read_file('optima_control_test.pickle',OPTIMA_CONTROL_DIR_TEST)
        optima_dem_test     = read_file('optima_dem_test.pickle',    OPTIMA_DEMENTIA_DIR_TEST)
        return dbank_control_test, dbank_dem_test, optima_control_test, optima_dem_test
    elif dataset == 'all':
        dbank_control_all  = read_file('dbank_control_all.pickle', DEMENTIABANK_CONTROL_DIR_ALL)
        dbank_dem_all      = read_file('dbank_dem_all.pickle',     DEMENTIABANK_DEMENTIA_DIR_ALL)
        optima_control_all = read_file('optima_control_all.pickle',OPTIMA_CONTROL_DIR_ALL)
        optima_dem_all     = read_file('optima_dem_all.pickle',    OPTIMA_DEMENTIA_DIR_ALL)
        return dbank_control_all, dbank_dem_all, optima_control_all, optima_dem_all
    else:
        print "Error in get_data(dataset): dataset must be one of 'all','train' or 'test'"
        return 


def get_all_book_pickles():
    wtow           = get_book_data('wtow.pickle',   WTOW_DIR)
    ijamob         = get_book_data('ijamob.pickle', IJAMOB_DIR)
    return wtow, ijamob

def get_dbank_control():
    return get_data('dbank_control.pickle',DEMENTIABANK_CONTROL_DIR_ALL)

def get_dbank_dem():
    return get_data('dbank_dem.pickle', DEMENTIABANK_DEMENTIA_DIR_ALL)

# -------------End Of Get Data functions-----------


# -------------Extract feature methods -----------
def extract_features(data,pickle_name = None, pickle_frequency = 20):
    feature_set = []
    # Load partial pickle if exists
    # We do this so if something crashes midrun we don't need to rerun the whole feature extractor
    if pickle_name and os.path.exists(OUTPUT_DIR + pickle_name):
        print "Partial pickle found at:", OUTPUT_DIR + pickle_name
        feature_set = open_pickle(OUTPUT_DIR + pickle_name)

    if len(feature_set) == len(data):
        print pickle_name, "features loaded"
    else:
        print "Loading and continuing from interview:", len(feature_set)
        # Continue extracting from data[len(feature_set):] (will be zero if no pickle found)
        for idx, interview in enumerate(data[len(feature_set):]):
            if len(interview) == 0:
                continue
            print "Extracting features for interview: ", len(feature_set) + 1 
            feat_dict  = pos_phrases.get_all(interview)
            feat_dict.update(pos_syntactic.get_all(interview))
            feat_dict.update(psycholinguistic.get_all(interview))
            feature_set.append(feat_dict)

            # Save every pickle_frequency'th interview 
            if pickle_name and idx % pickle_frequency == 0:
                print "Saving interview feature vector up to:", len(feature_set)
                overwrite_pickle(OUTPUT_DIR + pickle_name, feature_set)
    return feature_set


def make_feature_vec_pickles(dataset, picklename):
    print "=========================="
    print "Making feature vector for:", picklename
    feature_vecs = extract_features(dataset,picklename,10)
    # Make iid 
    random.shuffle(feature_vecs)
    overwrite_pickle(OUTPUT_DIR + picklename, feature_vecs)
    print picklename, "complete"
    print "=========================="

# -------------End of extract feature methods -----------

# ---------------- Get feature vectors -----------------#

def get_clinical_feature_data():
    optimal_dem = open_pickle(OUTPUT_DIR + "optima_dem_feature_vector.pickle")
    labels = ['Dementia'] * len(optimal_dem)
    optimal_con = open_pickle(OUTPUT_DIR + "optima_control_feature_vector.pickle")
    labels.extend(["Control"] * len(optimal_con))
    dementia_dem = open_pickle(OUTPUT_DIR + "dbank_dem_feature_vector.pickle")
    labels.extend(["Dementia"] * len(dementia_dem))
    dementia_con = open_pickle(OUTPUT_DIR + "dbank_control_feature_vector.pickle")
    labels.extend(["Control"] * len(dementia_con))
    data = []
    data.extend(optimal_dem)
    data.extend(optimal_con)
    data.extend(dementia_dem)
    data.extend(dementia_con)

    data = zip(data, labels)
    return data


def get_optima_feature_data():
    optimal_dem = open_pickle(OUTPUT_DIR + "optima_dem_feature_vector.pickle")
    labels = ['Dementia'] * len(optimal_dem)

    optimal_con = open_pickle(OUTPUT_DIR + "optima_control_feature_vector.pickle")
    labels.extend(["Control"] * len(optimal_con))

    data = []
    data.extend(optimal_dem)
    data.extend(optimal_con)
    data = zip(data, labels)
    return data


def get_dementiabank_feature_data():
    dementia_dem = open_pickle(OUTPUT_DIR + "dbank_dem_feature_vector.pickle")
    labels = ["Dementia"] * len(dementia_dem)

    dementia_con = open_pickle(OUTPUT_DIR + "dbank_control_feature_vector.pickle")
    labels.extend(["Control"] * len(dementia_con))

    data = []
    data.extend(dementia_dem)
    data.extend(dementia_con)
    data = zip(data, labels)
    return data


def get_all_feature_data():
    clinical_data = get_clinical_feature_data()
    ijamob_data = open_pickle(OUTPUT_DIR + "ijamob.pickle")
    wtow_data = open_pickle(OUTPUT_DIR + "wtow.pickle")
    labels = ['Dementia'] * len(wtow_data)
    labels.extend(['Control'] * len(ijamob_data))
    non_clinical_data = []
    non_clinical_data.extend(wtow_data)
    non_clinical_data.extend(ijamob_data)
    non_clinical_data = zip(non_clinical_data, labels)
    data = []
    data.extend(clinical_data)
    data.extend(non_clinical_data)
    return data



if __name__ == '__main__':

    # # Check if feature vector pickles exist - if so use them, if not parse
    # parser.parse(TEST_DIR)
    dbank_control_train, dbank_dem_train, optima_control_train, optima_dem_train = get_data("train")
    dbank_control_test,  dbank_dem_test,  optima_control_test,  optima_dem_test  = get_data("test")
    # wtow, ijamob = get_all_book_pickles()

    # Load and pickle dbank_dem
    make_feature_vec_pickles(dbank_dem_train,"dbank_dem_fv_train.pickle")
    make_feature_vec_pickles(dbank_dem_test,"dbank_dem_fv_test.pickle")
    
    # Load and pickle dbank_control
    make_feature_vec_pickles(dbank_control_train,"dbank_control_fv_train.pickle")
    make_feature_vec_pickles(dbank_control_test,"dbank_control_fv_test.pickle")
    
    # Load and pickle optima_control
    make_feature_vec_pickles(optima_control_train,"optima_control_fv_train.pickle")
    make_feature_vec_pickles(optima_control_test,"optima_control_fv_test.pickle")
    
    # Load and pickle optima_dem
    make_feature_vec_pickles(optima_dem_train,"optima_dem_fv_train.pickle")
    make_feature_vec_pickles(optima_dem_test,"optima_dem_fv_test.pickle")

    # # Load and pickle wtow
    # make_feature_vec_pickles(wtow, "wtow.pickle")

    # # Load and pickle ijamob
    # make_feature_vec_pickles(ijamob, "ijamob.pickle")

