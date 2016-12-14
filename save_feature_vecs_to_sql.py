from sqlalchemy import create_engine
import pandas as pd
from FeatureExtractor import pos_phrases 
from FeatureExtractor import pos_syntactic 
from FeatureExtractor import psycholinguistic
from FeatureExtractor import acoustic
from FeatureExtractor import discourse
from FeatureExtractor import parser
import pickle 

# ======================
# setup mysql connection
# ----------------------
USER   = 'dementia'
PASSWD = 'Dementia123!'
DB     = 'dementia'
url    = 'mysql://%s:%s@127.0.0.1/%s' % (USER, PASSWD, DB) 
engine = create_engine(url)
cnx    = engine.connect()
# ======================

PICKLE_PATH = 'data/pickles/data.pkl'

def save_data_to_pickle():
    data = parser.parse(['data/dbank/dementia/','data/dbank/control/'])
    output = open(PICKLE_PATH, 'wb')
    pickle.dump(data, output)
    output.close()


def get_parsed_data_from_pickle():
    print "Retriving data from %s" % PICKLE_PATH
    pkl_file = open(PICKLE_PATH, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def save_lexical():
    data = get_parsed_data_from_pickle()
    frames = []
    for interview in data:
        print "Processing: %s" % interview
        # Make feature dictionary
        feat_dict = pos_phrases.get_all(data[interview])
        feat_dict.update(pos_syntactic.get_all(data[interview]))
        feat_dict.update(psycholinguistic.get_psycholinguistic_features(data[interview]))

        # Save to sql, match interview name with acoustic files
        feat_dict['interview'] = interview.replace(".txt",'c')
        feat_df = pd.DataFrame([feat_dict])
        frames.append(feat_df)
    
    # Merge and reset index  
    feat_df = pd.concat(frames)
    feat_df.reset_index(inplace=True)

    # Save to database
    feat_df.drop('index', axis=1, inplace=True)
    feat_df.to_sql("dbank_lexical", cnx, if_exists='replace', index=False)


def save_cookie_theft_info_units():
    data = get_parsed_data_from_pickle()
    frames = []
    for interview in data:
        print "Processing: %s" % interview
        # Make feature dictionary
        feat_dict = psycholinguistic.get_cookie_theft_info_unit_features(data[interview])
        # Save to sql, match interview name with acoustic files
        feat_dict['interview'] = interview.replace(".txt",'c')
        feat_df = pd.DataFrame([feat_dict])
        frames.append(feat_df)
    
    # Merge and reset index  
    feat_df = pd.concat(frames, ignore_index=True)
    # Save to database
    feat_df.to_sql("dbank_cookie_theft_info_units", cnx, if_exists='replace', index=False)


def save_spatial():
    data = get_parsed_data_from_pickle()
    halves_frame = []
    strips_frame = []
    quadrants_frame = []
    for interview in data:
        print "Processing: %s" % interview
        # halves            
        feat_dict = psycholinguistic.get_spatial_features(data[interview], "halves")
        feat_dict['interview'] = interview.replace(".txt",'c')
        feat_df = pd.DataFrame([feat_dict])
        halves_frame.append(feat_df)
        # strips    
        feat_dict = psycholinguistic.get_spatial_features(data[interview], "strips")
        feat_dict['interview'] = interview.replace(".txt",'c')
        feat_df = pd.DataFrame([feat_dict])
        strips_frame.append(feat_df)
        # quadrants    
        feat_dict = psycholinguistic.get_spatial_features(data[interview], "quadrants")
        feat_dict['interview'] = interview.replace(".txt",'c')
        feat_df = pd.DataFrame([feat_dict])
        quadrants_frame.append(feat_df)

    halves_df    = pd.concat(halves_frame, ignore_index=True)
    strips_df    = pd.concat(strips_frame, ignore_index=True)
    quadrants_df = pd.concat(quadrants_frame, ignore_index=True)

    halves_df.to_sql("dbank_spatial_halves", cnx, if_exists='replace', index=False)
    strips_df.to_sql("dbank_spatial_strips", cnx, if_exists='replace', index=False)
    quadrants_df.to_sql("dbank_spatial_quadrants", cnx, if_exists='replace', index=False)


def save_acoustic():
    # Extract control acoustic features
    control  = acoustic.get_all("data/soundfiles/control")
    df_control = pd.DataFrame.from_dict(control, orient="index")
    df_control['control'] = True
    
    # Extract dementia acoustic features
    dementia  = acoustic.get_all("data/soundfiles/dementia")
    df_dementia = pd.DataFrame.from_dict(dementia, orient="index")
    df_dementia['control'] = False
    
    # Merge dfs
    feat_df = pd.concat([df_dementia, df_control])

    # Save interview field for joins 
    feat_df["interview"] = feat_df.index
    feat_df.reset_index(inplace=True)

    # Save to sql  
    feat_df.to_sql("dbank_acoustic_lemmetized", cnx, if_exists='replace')


def save_discourse():
    # Extract control discourse features
    control  = discourse.get_all("data/dbank/discourse_trees/control")
    df_control = pd.DataFrame.from_dict(control, orient="index")
    
    # Extract dementia discourse features
    dementia  = discourse.get_all("data/dbank/discourse_trees/dementia")
    df_dementia = pd.DataFrame.from_dict(dementia, orient="index")
    
    # Merge dfs
    feat_df = pd.concat([df_dementia, df_control])

    # Save interview field for joins 
    feat_df["interview"] = feat_df.index
    feat_df['interview'] = feat_df['interview'] + 'c' # So it's consistent with sound files

    # Save to sql  
    feat_df.to_sql("dbank_discourse", cnx, if_exists='replace', index=False)


def debug():
    data = get_parsed_data_from_pickle()
    frames = []
    for interview in data:
        print "Processing: %s" % interview
        # Make feature dictionary
        feat_dict = pos_phrases.get_all(data[interview])
        feat_dict.update(pos_syntactic.get_all(data[interview]))
        feat_dict.update(psycholinguistic.get_cookie_theft_info_unit_features(data[interview]))
        feat_dict.update(psycholinguistic.get_psycholinguistic_features(data[interview]))
        # feat_dict.update(psycholinguistic.get_spatial_features(data[interview], 'halves'))
        # feat_dict.update(psycholinguistic.get_spatial_features(data[interview], 'strips'))
        # feat_dict.update(psycholinguistic.get_spatial_features(data[interview], 'quadrants'))
        import pdb; pdb.set_trace()
        # Save to sql, match interview name with acoustic files
        feat_dict['interview'] = interview.replace(".txt",'c')
        feat_df = pd.DataFrame([feat_dict])
        frames.append(feat_df)
    
if __name__ == '__main__':
    save_spatial()
    # save_cookie_theft_info_units()
