from sqlalchemy import create_engine
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.metrics import f1_score

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


# ------------------
# Helper functions
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# One time use to clean up sql table
# def clean_sql():
#     lexical = pd.read_sql_table("dbank_lexical", cnx)
#     acoustic = pd.read_sql_table("dbank_acoustic", cnx)

#     lexical['control'] = lexical['control'].astype('bool')
#     acoustic['control'] = acoustic['control'].astype('bool')

#     lexical['dementia'] = ~lexical['control']
#     acoustic['dementia'] = ~acoustic['control']

#     lexical = lexical.drop(['index', 'control'], 1)
#     acoustic = acoustic.drop(['index', 'control'], 1)

#     # Save as different name to prevent original tables from being deleted if there's an error
#     lexical.to_sql("dbank_lexical_clean", cnx, if_exists='replace', index=False)
#     acoustic.to_sql("dbank_acoustic_clean", cnx, if_exists='replace', index=False)
# ------------------


def get_data():
    # Read from sql
    # (indexes are not synced so drop'em)
    lexical  = pd.read_sql_table("dbank_lexical", cnx)
    acoustic = pd.read_sql_table("dbank_acoustic", cnx)
    
    # Merge
    fv = pd.merge(lexical, acoustic, on=['interview', 'dementia'])

    # Shuffle
    fv = fv.sample(frac=1, random_state=20).reset_index()

    # Split 
    y = fv['dementia'].astype('bool')
    X = fv.drop(['dementia', 'interview', 'level_0', 'index'], 1)

    # Return
    return X, y 


def feature_selection(X,y,nfeat):
    # Feature selection
    fit = SelectKBest(f_classif, k=nfeat).fit(X, y)
    X_fs = fit.transform(X)
    print "%d features removed" % (X.shape[1] - X_fs.shape[1])
    print("Selected Features: %s") % X.columns.values[fit.get_support()]    
    return X_fs


def evaluate(model, X, y):
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)

    # Fit
    model.fit(X_train, y_train)
    
    # Predict
    yhat = model.predict(X_test)

    # Print report
    print(f1_score(y_test, yhat))


if __name__ == '__main__':
    X, y = get_data()
    X_fs = feature_selection(X, y, 60)
    evaluate(linear_model.LogisticRegression(), X_fs, y)


