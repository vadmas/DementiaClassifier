# Small script which took a pickle from a previous version and saved using dataframes
# (one-use script because we no longer use dbank_control.pickle)

import pandas as pd
# Use cPickle if available
try:
    import cPickle as pickle
except:
    import pickle

# Helper function
def open_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def to_dataframe(interview, name, control):
    df = pd.DataFrame(interview)
    df['control']   = control
    df['utterance'] = df.index
    df['interview_code'] = name
    return df
# ----------------------
# Load dementia bank data 
# ----------------------
dbank_control_path = "data/pickles/dbank_control.pickle"
dbank_dem_path = "data/pickles/dbank_dem.pickle"

print "Loading pickle from : %s..." % dbank_control_path
control = open_pickle(dbank_control_path)
print "Loading pickle from : %s..." % dbank_dem_path
dem = open_pickle(dbank_dem_path)

c_keys = control.keys()
d_keys = dem.keys()
c_keys.sort()
d_keys.sort()


# ----------------------
# Save to pandas
# ----------------------

frames = []

for key in c_keys:
    frames.append(to_dataframe(control[key], key, True))

for key in d_keys:
    frames.append(to_dataframe(dem[key], key, False))

dfs = pd.concat(frames)
dfs.to_pickle("data/pickles/dbank.pickle")
