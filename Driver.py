# Use cPickle if available
try:
    import cPickle as pickle
except:
    import pickle
import parser
import os
from FeatureExtractor import parser 
# from FeatureExtractor import pos_phrases 
from FeatureExtractor import pos_syntactic 
#from FeatureExtractor import psycholinguistic

# constants
DEMENTIABANK_CONTROL_DIR  = 'data/processed/dbank/control'
DEMENTIABANK_DEMENTIA_DIR = 'data/processed/dbank/dementia'
OPTIMA_CONTROL_DIR        = 'data/processed/optima/nometa/control'
OPTIMA_DEMENTIA_DIR       = 'data/processed/optima/nometa/dementia'

#Welcome To Our World text file
WTOW_DIR                  = 'data/processed/optima/nometa/dementia'

#It's Just a Matter Of Balance text file
IJAMOB_DIR                = 'data/processed/optima/nometa/dementia'
PICKLE_DIR                = 'data/pickles/'
TEST_DIR                  = 'data/test/'

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

if __name__ == '__main__':
    dbank_control, dbank_dem, optima_control, optima_dem = get_all_pickles()
    # data = parser.parse(TEST_DIR)
    # f1 = phrases.get_all_features(dbank_control)
    # f2 = psycholinguistic.get_all_features(dbank_control)
    # f3 = syntactic.get_all_features(dbank_control)

    # pos_syntactic_extractor.get_structure_features(dbank_control)
    # psycholinguistic.get_all_features(dbank_control)

    # print "DBank Control: "  + str(len(dbank_control))
    # print "DBank Dem: " 	 + str(len(dbank_dem))
    # print "Optima Control: " + str(len(optima_control))
    # print "Optima Dem: "	 + str(len(optima_dem))
