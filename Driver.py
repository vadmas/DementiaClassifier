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

from random import shuffle

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

#Processed feature vector directory
FEATURE_VEC_DIR = "FeatureVecs/"

#Arff file directory
ARFF_DIR = 'arff files/'

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

def get_all_book_pickles():
    wtow           = get_book_data('wtow.pickle',   WTOW_DIR)
    ijamob         = get_book_data('ijamob.pickle', IJAMOB_DIR)
    return wtow, ijamob

# -------------End Of Get Data functions-----------


# -------------Extract feature methods -----------
def extract_features(data,pickle_name = None, pickle_frequency = 20):
    feature_set = []
    # Load partial pickle if exists
    # We do this so if something crashes midrun we don't need to rerun the whole feature extractor
    if pickle_name and os.path.exists(FEATURE_VEC_DIR + pickle_name):
        print "Partial pickle found at:", FEATURE_VEC_DIR + pickle_name
        feature_set = open_pickle(FEATURE_VEC_DIR + pickle_name)

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
                overwrite_pickle(FEATURE_VEC_DIR + pickle_name, feature_set)
    return feature_set


def make_feature_vec(picklename, dataset):
    print "Making feature vector for:", picklename
    feature_vecs = extract_features(dataset,picklename,10)
    # Make iid 
    shuffle(feature_vecs)
    overwrite_pickle(FEATURE_VEC_DIR + picklename, feature_vecs)
    print picklename, "complete"
    print "--------------------------"
    return feature_vecs

# -------------End of extract feature methods -----------

# ---------------- Make arff files -----------------#

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

def add_labels(data,label):
    labels = [label] * len(data)
    data = zip(data, labels)
    return data

def make_arff_file(samples, file_name):
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
        key = str(k).strip().replace('\'', '')
        attribute_str += '\'' + key + '\''+ ' ' + get_attribute_from_variable(v)
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


if __name__ == '__main__':

    print ''' 
        =============================
        Getting raw text dictionaries
        ============================= 
    '''
    dbank_control_train  = read_file('dbank_control_train.pickle', DEMENTIABANK_CONTROL_DIR_TRAIN)
    dbank_control_test   = read_file('dbank_control_test.pickle',  DEMENTIABANK_CONTROL_DIR_TEST)

    dbank_dem_train      = read_file('dbank_dem_train.pickle',     DEMENTIABANK_DEMENTIA_DIR_TRAIN)
    dbank_dem_test       = read_file('dbank_dem_test.pickle',      DEMENTIABANK_DEMENTIA_DIR_TEST)

    optima_control_train = read_file('optima_control_train.pickle',OPTIMA_CONTROL_DIR_TRAIN)
    optima_control_test  = read_file('optima_control_test.pickle', OPTIMA_CONTROL_DIR_TEST)

    optima_dem_train     = read_file('optima_dem_train.pickle',    OPTIMA_DEMENTIA_DIR_TRAIN)
    optima_dem_test      = read_file('optima_dem_test.pickle',     OPTIMA_DEMENTIA_DIR_TEST)

    print ''' 
        =======================================
        Turning into numerical feature vectors 
        ======================================= 
    '''
    # Load and pickle dbank_dem
    dbank_dem_train_fv      = make_feature_vec("dbank_dem_fv_train.pickle", dbank_dem_train)
    dbank_dem_test_fv       = make_feature_vec("dbank_dem_fv_test.pickle", dbank_dem_test)
    
    # Load and pickle dbank_control
    dbank_control_train_fv  = make_feature_vec("dbank_control_fv_train.pickle", dbank_control_train)
    dbank_control_test_fv   = make_feature_vec("dbank_control_fv_test.pickle", dbank_control_test)
    
    # Load and pickle optima_control
    optima_control_train_fv = make_feature_vec("optima_control_fv_train.pickle", optima_control_train)
    optima_control_test_fv  = make_feature_vec("optima_control_fv_test.pickle", optima_control_test)
    
    # Load and pickle optima_dem
    optima_dem_train_fv     = make_feature_vec("optima_dem_fv_train.pickle", optima_dem_train)
    optima_dem_test_fv      = make_feature_vec("optima_dem_fv_test.pickle", optima_dem_test)
    
    print ''' 
        ==================
        Making arff files 
        ================== 
    '''
    make_arff_file(add_labels(dbank_dem_train_fv,"Dementia"),     "dbank_dem_train")
    make_arff_file(add_labels(dbank_dem_test_fv, "Dementia"),     "dbank_dem_test")
    make_arff_file(add_labels(dbank_control_train_fv,"Control"),  "dbank_control_train")
    make_arff_file(add_labels(dbank_control_test_fv, "Control"),  "dbank_control_test")
    make_arff_file(add_labels(optima_control_train_fv,"Control"), "optima_control_train")
    make_arff_file(add_labels(optima_control_test_fv, "Control"), "optima_control_test")
    make_arff_file(add_labels(optima_dem_train_fv,"Dementia"),    "optima_dem_train")
    make_arff_file(add_labels(optima_dem_test_fv, "Dementia"),    "optima_dem_test")

    print "Done! Arff files located at: " + ARFF_DIR

