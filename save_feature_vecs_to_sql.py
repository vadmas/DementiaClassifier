from sqlalchemy import create_engine
import pandas as pd
from FeatureExtractor import pos_phrases 
from FeatureExtractor import pos_syntactic 
from FeatureExtractor import psycholinguistic
from FeatureExtractor import acoustic

# ======================
# setup mysql connection
# ----------------------
USER   = 'dementia'
PASSWD = 'Dementia123!'
DB     = 'dementia'
url = 'mysql://%s:%s@localhost/%s' % (USER, PASSWD, DB) 
engine = create_engine(url)
cnx = engine.connect()
# ======================


def save_lexical():
    df = pd.read_pickle("data/pickles/dbank.pickle")
    interviews = pd.unique(df.interview_code.ravel()).tolist()

    frames = []
    for i in interviews:
        # Make interview dict from dataframe
        print "Processing: %s" % i
        interview = df.loc[df['interview_code'] == i][['basic_dependencies','parse_tree','pos','pos_freq','raw', 'token','control']]
        control   = pd.unique(interview.control.ravel()).tolist()
        interview = interview.drop('control',1)
        interview = interview.to_dict('records')

        # Make feature dictionary
        feat_dict = pos_phrases.get_all(interview)
        feat_dict.update(pos_syntactic.get_all(interview))
        feat_dict.update(psycholinguistic.get_all(interview))

        # Save to sql
        feat_dict['control'] = control
        feat_dict['interview'] = i.replace(".txt",'')
        feat_df = pd.DataFrame(feat_dict)

        frames.append(feat_df)
    
    # Merge and reset index  
    feat_df = pd.concat(frames)
    feat_df.reset_index(inplace=True)

    # Save to database
    feat_df.to_sql("dbank_lexical", cnx, if_exists='replace')


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
    feat_df.to_sql("dbank_acoustic", cnx, if_exists='replace')

if __name__ == '__main__':
    # save_acoustic()
    # save_lexical()
