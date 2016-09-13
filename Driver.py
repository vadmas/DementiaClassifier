from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import LabelKFold
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# models
from sklearn import linear_model
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

from FeatureExtractor.fraser_feature_set import get_top_50

# hide depreciation warning (sklearn)
# ====================
import warnings
warnings.filterwarnings('ignore')
# ====================
# globals
XTICKS = np.arange(30,150,3)
REGULARIZATION_CONSTANT = 1
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

ALZHEIMERS = ["PossibleAD", "ProbableAD"]
CONTROL = ['Control']

# ======================
# setup mysql connection
# ----------------------
USER   = 'dementia'
PASSWD = 'Dementia123!'
DB     = 'dementia'
# url = 'mysql://%s:%s@127.0.0.1/%s' % (USER, PASSWD, DB) 
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


# # one-time use function to fix bugs in sql table
# def update_sql():
#     lexical = pd.read_sql_table("dbank_lexical", cnx)
#     lexical_new = pd.read_sql_table("dbank_lexical_new", cnx)
#     import pdb; pdb.set_trace()
#     print "test"


def plot_results(results_arr, names_arr, xlabel, ylabel):

    if len(results_arr) != len(names_arr):
        print "error: Results and names array not same length"
        return

    means  = []
    stdevs = []

    for results in results_arr:
        arr = np.asarray(results)
        means.append(np.mean(arr, axis=0))
        stdevs.append(np.std(arr, axis=0))

    df_dict  = {}
    err_dict = {}

    title = "Model comparison: "
    for i, name in enumerate(names_arr):
        max_str = "%s max: %f" % (names_arr[i], means[i].max())
        print max_str
        title += max_str
        df_dict[name] = means[i]
        err_dict[name] = stdevs[i]

    df = pd.DataFrame(df_dict)
    df.index = XTICKS
    plot = df.plot(yerr=err_dict, title=title)
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)
    plt.show()


def get_top_pearson_features(X,y,n):
    df = pd.DataFrame(X).apply(pd.to_numeric)
    df['y'] = y
    corr_coeff = df.corr()['y'].abs().sort(inplace=False, ascending=False)
    return corr_coeff.index.values[1:n+1].astype(int)


def get_data(lexical_table_name):
    # Read from sql
    lexical  = pd.read_sql_table(lexical_table_name, cnx)
    acoustic = pd.read_sql_table("dbank_acoustic", cnx)
    diag     = pd.read_sql_table("diagnosis", cnx)
    
    # Merge
    fv = pd.merge(lexical, acoustic, on=['interview'])

    # # Remove non-AD samples
    diag = diag[diag['diagnosis'].isin(ALZHEIMERS + CONTROL)]
    fv = pd.merge(fv,diag)
    
    # Randomize
    fv = fv.sample(frac=1,random_state=20)

    # Collect Labels 
    labels = [label[:3] for label in fv['interview']]
    
    # Split 
    y = fv['dementia'].astype('bool')
    X = fv.drop(['dementia', 'interview', 'level_0', 'diagnosis'], 1)

    # Return
    return X, y, labels


def fraser_features(X):
    fraser = get_top_50()
    return X[fraser]


def evaluate_model(model, lex_name):
    X, y, labels = get_data(lex_name)
    
    # Split into folds using labels 
    label_kfold = LabelKFold(labels, n_folds=10)

    folds  = []
    
    # Feature analysis
    # columns = X.columns 
    # feat_tally = defaultdict(int)
    # z = 0.0

    for train_index, test_index in label_kfold:
        print "processing fold: %d" % (len(folds) + 1)
        
        # Split
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        scores = []     
        for k in XTICKS:
            # Perform feature selection
            # fit = SelectKBest(f_classif, k=k).fit(X_train, y_train)
            # indices = fit.get_support(indices=True)

            indices = get_top_pearson_features(X_train, y_train, k)
            
            # # Feature analysis
            # for i in indices: 
            #     feat_tally[i] += 1
            # z += 1

            # Select k features 
            X_train_fs = X_train[:,indices]
            X_test_fs  = X_test[:,indices]
            
            # Fit    
            model.fit(X_train_fs, y_train)
            
            # Predict
            yhat  = model.predict(X_test_fs)           
            # Save
            scores.append(accuracy_score(y_test, yhat))
            
           
        # ----- save row ----- 
        folds.append(scores)
        # -------------------- 

    # feature_scores = [(val/z, columns[key]) for key, val in feat_tally.iteritems()]
    # feature_scores.sort(reverse=True)
    # for fs in feature_scores:
    #     print "Score: %f, %s " % fs

    folds = np.asarray(folds)
    return folds 


if __name__ == '__main__':
    # update_sql()
    model = linear_model.LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)
    dbank_lexical                     = evaluate_model(model, "dbank_lexical_new2")
    dbank_lexical_sentence_disfluency = evaluate_model(model, "dbank_lexical_sentence_disfluency")
    dbank_lexical_sentence            = evaluate_model(model, "dbank_lexical_sentence")
    # dbank_lexical_lemmetized          = evaluate_model(model, "dbank_lexical_lemmetized")
    
    plot_results([dbank_lexical, dbank_lexical_sentence_disfluency,dbank_lexical_sentence,], 
                 ["dbank_lexical", "dbank_lexical_sentence_disfluency", "dbank_lexical_sentence",],
                  "# of Features",
                  "Accuracy ")
