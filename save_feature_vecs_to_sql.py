from sqlalchemy import create_engine
import pandas as pd
from FeatureExtractor import pos_phrases 
from FeatureExtractor import pos_syntactic 
from FeatureExtractor import psycholinguistic
from FeatureExtractor import acoustic
from FeatureExtractor.parser import parse


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


def save_lexical(data):
    # df = pd.DataFrame.from_dict(data)
    # interviews = pd.unique(df.interview_code.ravel()).tolist()
    frames = []
    for interview in data:
        print "Processing: %s" % interview
        # Make feature dictionary
        feat_dict = pos_phrases.get_all(data[interview])
        feat_dict.update(pos_syntactic.get_all(data[interview]))
        feat_dict.update(psycholinguistic.get_all(data[interview]))

        # Save to sql, match interview name with acoustic files
        feat_dict['interview'] = interview.replace(".txt",'c')
        feat_df = pd.DataFrame([feat_dict])
        frames.append(feat_df)
    
    # Merge and reset index  
    feat_df = pd.concat(frames)
    feat_df.reset_index(inplace=True)

    # Save to database
    feat_df.drop('index', axis=1, inplace=True)
    feat_df.to_sql("dbank_lexical_lemmetized", cnx, if_exists='replace', index=False)


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


if __name__ == '__main__':
    data = parse(['data/dbank/dementia','data/dbank/control'])
    save_lexical(data)
