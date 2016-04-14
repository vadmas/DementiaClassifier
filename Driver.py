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

#PICKLE_DIR                = 'data/pickles/'
PICKLE_DIR                 = 'stanford/processed/pickles/'
TEST_DIR                  = 'data/test/'

#Output Directory
OUTPUT_DIR="FeatureVecs/"

#Check pickle first, use parser if pickle doesn't exist
def get_data(picklename, raw_files_directory):
    if os.path.exists(PICKLE_DIR + picklename):
        print "Pickle found at: " + PICKLE_DIR + picklename
        with open(PICKLE_DIR + picklename, 'rb') as handle:
            data = pickle.load(handle)
    else:
        print "Pickle not found, beginning parse."
        data = parser.parse(raw_files_directory)
        with open(PICKLE_DIR + picklename, 'wb') as handle:
            pickle.dump(data, handle)
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

def extract_features(data):
    feature_set = []
    for idx, interview in enumerate(data):
        print "Extracting features for interview: ", idx
        feat_dict  = pos_phrases.get_all(interview)
        feat_dict.update(pos_syntactic.get_all(interview))
        feat_dict.update(psycholinguistic.get_all(interview))
        feature_set.append(feat_dict)
    return feature_set

if __name__ == '__main__':
    
    # Check if feature vector pickles exist - if so use them, if not parse
    dbank_control, dbank_dem, optima_control, optima_dem = get_all_pickles()

    # Load and pickle dementiabank control
    dbank_control_vec = extract_features(dbank_control)
    with open(OUTPUT_DIR + "dbank_control_vec.pickle", 'wb') as handle:
            pickle.dump(dbank_control_vec, handle)

    # Load and pickle dementiabank dementia
    dbank_dem_vec = extract_features(dbank_dem)
    with open(OUTPUT_DIR + "dbank_dem_vec.pickle", 'wb') as handle:
        pickle.dump(dbank_dem_vec, handle)

    # Load and pickle optima control
    # optima_control_vec = extract_features(optima_control)
    # with open(OUTPUT_DIR + "optima_control_vec.pickle", 'wb') as handle:
    #         pickle.dump(optima_control_vec, handle)
    
    # # Load and pickle optima dementia
    # optima_dem_vec = extract_features(optima_dem)
    # with open(OUTPUT_DIR + "optima_dem_vec.pickle", 'wb') as handle:
    #         pickle.dump(optima_dem_vec, handle)
