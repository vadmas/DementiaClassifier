# Use cPickle if available
try:
    import cPickle as pickle
except:
    import pickle
import parser
import os
from FeatureExtractor import parser 
from FeatureExtractor import pos_phrases 
from FeatureExtractor import pos_syntactic 
from FeatureExtractor import psycholinguistic

# constants
DEMENTIABANK_CONTROL_DIR  = 'data/processed/dbank/control'
DEMENTIABANK_DEMENTIA_DIR = 'data/processed/dbank/dementia'
OPTIMA_CONTROL_DIR        = 'data/processed/optima/nometa/control'
OPTIMA_DEMENTIA_DIR       = 'data/processed/optima/nometa/dementia'

#Welcome To Our World text file
WTOW_DIR                  = 'data/processed/wtow'

#It's Just a Matter Of Balance text file
IJAMOB_DIR                = 'data/processed/ijamob'

PICKLE_DIR                = 'data/pickles/'
TEST_DIR                  = 'data/test/'

#Output Directory
OUTPUT_DIR="FeatureVecs/"

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

def get_data(picklename, raw_files_directory):
    #Check pickle first, use parser if pickle doesn't exist
    if os.path.exists(PICKLE_DIR + picklename):
        print "Pickle found at: " + PICKLE_DIR + picklename
        data = open_pickle(PICKLE_DIR + picklename)
    else:
        print "Pickle not found, beginning parse."
        data = parser.parse(raw_files_directory)
        save_pickle(PICKLE_DIR + picklename,data)
    return data

def get_all_pickles():
    dbank_control  = get_data('dbank_control.pickle', DEMENTIABANK_CONTROL_DIR)
    dbank_dem      = get_data('dbank_dem.pickle',     DEMENTIABANK_DEMENTIA_DIR)
    optima_control = get_data('optima_control.pickle',OPTIMA_CONTROL_DIR)
    optima_dem     = get_data('optima_dem.pickle',    OPTIMA_DEMENTIA_DIR)
    return dbank_control, dbank_dem, optima_control, optima_dem

def get_dbank_control():
    return get_data('dbank_control.pickle',DEMENTIABANK_CONTROL_DIR)

def get_dbank_dem():
    return get_data('dbank_dem.pickle', DEMENTIABANK_DEMENTIA_DIR)

# -------------End Of Get Data functions-----------


# -------------Extract feature methods -----------
def extract_features(data,pickle_name = None, pickle_frequency = 20):
    feature_set = []
    # Load partial pickle if exists
    # We do this so if something crashes midrun we don't need to rerun the whole feature extractor
    if pickle_name and os.path.exists(OUTPUT_DIR + pickle_name):
        print "Partial pickle found at:", OUTPUT_DIR + pickle_name
        feature_set = open_pickle(OUTPUT_DIR + pickle_name)
        print "Loading and continuing from interview:", len(feature_set)

    # Continue extracting from data[len(feature_set):] (will be zero if no pickle found)
    for idx, interview in enumerate(data[len(feature_set):]):
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
    feature_vecs = extract_features(dataset,picklename,10)
    overwrite_pickle(picklename, feature_vecs)

# -------------End of extract feature methods -----------

if __name__ == '__main__':
    
    # # Check if feature vector pickles exist - if so use them, if not parse
    dbank_control, dbank_dem, optima_control, optima_dem = get_all_pickles()
    
    # Load and pickle dbank_control
    make_feature_vec_pickles(dbank_control,"dbank_control_feature_vector.pickle")

    # # Load and pickle dbank_dem
    # make_feature_vec_pickles(dbank_control,"dbank_dem_feature_vector.pickle")

    # # Load and pickle optima_control
    # make_feature_vec_pickles(dbank_control,"optima_control_feature_vector.pickle")

    # # Load and pickle optima_dem
    # make_feature_vec_pickles(dbank_control,"optima_dem_feature_vector.pickle")
