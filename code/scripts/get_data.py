from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import itertools
from FeatureExtractor import feature_sets
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle

ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ['Control']
NON_ALZHEIMERS = ["MCI", "Memory", "Other", "Vascular"]

# ------------------
# Diagnosis keys
# - Control
# - MCI
# - Memory
# - Other
# - PossibleAD
# - ProbableAD
# - Vascular
# ------------------

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


def get_data(spacial_db="dbank_spatial_halves", diagnosis=ALZHEIMERS+CONTROL, include_features=None, exclude_features=None, polynomial=True, with_disc=True, with_spatial=True):
    # Read from sql
    lexical    = pd.read_sql_table("dbank_lexical", cnx)
    acoustic   = pd.read_sql_table("dbank_acoustic", cnx)
    diag       = pd.read_sql_table("diagnosis", cnx)
    demo       = pd.read_sql_table("demographic_imputed", cnx)
    discourse  = pd.read_sql_table("dbank_discourse", cnx)
    infounits  = pd.read_sql_table("dbank_cookie_theft_info_units", cnx)
    spacial    = pd.read_sql_table(spacial_db, cnx)

    # Merge lexical and acoustic
    fv = pd.merge(lexical, acoustic, on=['interview'])

    # Add diagnosis
    diag = diag[diag['diagnosis'].isin(diagnosis)]
    fv = pd.merge(fv, diag)
    # Add demographics
    fv = pd.merge(fv, demo)
    # Add discourse
    if with_disc:
        fv = pd.merge(fv, discourse, on=['interview'])

    # Add infounits
    fv = pd.merge(fv, infounits, on=['interview'])

    # # Add spacial
    # (Add Polynomial)
    if with_spatial:
        if polynomial: 
            spacial = spacial.drop(exclude_features, errors='ignore')
            spacial = make_polynomial_terms(spacial)
        fv = pd.merge(fv, spacial, on=['interview'])

    # # Randomize
    fv = fv.sample(frac=1, random_state=20)

    # Collect Labels
    labels = [label[:3] for label in fv['interview']]
    # Split
    y = fv['dementia'].astype('bool')

    # Clean
    drop = ['dementia', 'level_0', 'interview', 'diagnosis', 'gender', 'index', 'gender_int']
    if exclude_features:
        drop += exclude_features
    X = fv.drop(drop, 1, errors='ignore')

    X = X.apply(pd.to_numeric, errors='ignore')

    if include_features:
        X = X[include_features]

    return X, y, labels


# Interaction groups specify which features should have interaction terms. Features within
# a group do not form interaction terms.
def make_polynomial_terms(df, keep_linear=True):
    cols = df.columns.drop('interview', errors='ignore')
    df = df.apply(pd.to_numeric, errors='ignore')

    for f1, f2 in itertools.combinations_with_replacement(cols, 2):
        if f1 == f2:
            prefix = 'sqr_'
        else:
            prefix = 'intr_'
        df[prefix + f1 + "_" + f2] = df[f1] * df[f2]

    if not keep_linear:
        df = df.drop(cols, axis=1, errors='ignore')
    return df


# one-time use function to fix bugs in sql table
def update_sql():
    lex_tmp = pd.read_sql_table("dbank_lexical_tmp", cnx)
    cookie  = pd.read_sql_table("dbank_cookie_theft_info_units", cnx)

    # lsrs    = pd.read_sql_table("leftside_rightside_polynomial", cnx)
    # halves  = pd.read_sql_table("dbank_spatial_halves", cnx)

    drop_rsls = ["count_of_leftside_keyword", "count_of_rightside_keyword", "leftside_keyword_to_non_keyword_ratio", "leftside_keyword_type_to_token_ratio", "percentage_of_leftside_keywords_mentioned", "percentage_of_rightside_keywords_mentioned",
                 "rightside_keyword_to_non_keyword_ratio", "rightside_keyword_type_to_token_ratio", "ratio_left_to_right_keyword_count", "ratio_left_to_right_keyword_to_word", "ratio_left_to_right_keyword_type_to_token", "ratio_left_to_right_keyword_percentage"]
    drop_cookie = cookie.columns.drop('interview')
    lex_tmp = lex_tmp.drop(drop_rsls,   axis=1, errors='ignore')
    lex_tmp = lex_tmp.drop(drop_cookie, axis=1, errors='ignore')
    # lex_tmp.to_sql("dbank_lexical", cnx, if_exists='replace', index=False)
    print "fin!"


def get_target_source_data(with_disc=True):
    ALZHEIMERS = ["PossibleAD", "ProbableAD"]
    CONTROL    = ['Control']
    MCI        = ["MCI"]
    
    to_exclude = feature_sets.get_general_keyword_features()

    # Get data
    X_alz, y_alz, l_alz = get_data(diagnosis=ALZHEIMERS, exclude_features=to_exclude, with_disc=with_disc)
    X_con, y_con, l_con = get_data(diagnosis=CONTROL,    exclude_features=to_exclude, with_disc=with_disc)
    X_mci, y_mci, l_mci = get_data(diagnosis=MCI,        exclude_features=to_exclude, with_disc=with_disc)

    # Split control samples into target/source set (making sure one patient doesn't appear in both t and s)
    gkf = GroupKFold(n_splits=6).split(X_con,y_con,groups=l_con)
    source, target = gkf.next()
    
    Xt, yt, lt = concat_and_shuffle(X_mci, y_mci, l_mci, X_con.ix[target], y_con.ix[target], np.array(l_con)[target])
    Xs, ys, ls = concat_and_shuffle(X_alz, y_alz, l_alz, X_con.ix[source], y_con.ix[source], np.array(l_con)[source])

    return Xt, yt, lt, Xs, ys, ls


def concat_and_shuffle(X1, y1, l1, X2, y2, l2, random_state=1):
    # Coerce all arguments to dataframes
    X1, X2 = pd.DataFrame(X1), pd.DataFrame(X2) 
    y1, y2 = pd.DataFrame(y1), pd.DataFrame(y2) 
    l1, l2 = pd.DataFrame(l1), pd.DataFrame(l2) 

    X_concat = X1.append(X2, ignore_index=True)
    y_concat = y1.append(y2, ignore_index=True)
    l_concat = l1.append(l2, ignore_index=True)
    
    X_shuf, y_shuf, l_shuf = shuffle(X_concat, y_concat, l_concat, random_state=random_state)

    return X_shuf, y_shuf, l_shuf[0].values.tolist()


