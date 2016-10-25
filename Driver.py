from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import LabelKFold
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt


# models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression

from FeatureExtractor.fraser_feature_set import get_top_50
from FeatureExtractor.domain_adaptation import expand_feature_space
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

# ====================
import warnings
warnings.filterwarnings('ignore')

# ====================
# globals
XTICKS = np.arange(20,75,1)
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

ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ['Control']
NON_ALZHEIMERS = ["MCI", "Memory", "Other", "Vascular"]

# # Hardcoded temporary feature list
# TOPSEVENTY = ["mfcc8_kurtosis","mfcc7_kurtosis","mfcc6_kurtosis","mfcc5_kurtosis","mfcc4_kurtosis","keywordIUSubjectWoman","keywordIUObjectWindow","keywordIUObjectStool","keywordIUObjectCurtains","getImagabilityScore","binaryIUSubjectWoman","binaryIUSubjectGirl","binaryIUPlaceExterior","binaryIUObjectWindow","binaryIUObjectStool","binaryIUObjectSink","binaryIUObjectCurtains","binaryIUActionStoolFalling","VP_to_VBG","RatioPronoun","PProportion","PP","NumAdverbs","NP_to_PRP","NP_to_DT_NN","MeanWordLength","AvgPPTypeLengthNonEmbedded","AvgPPTypeLengthEmbedded","ADVP","NumNouns","proportion_below_threshold_0.3","PPTypeRate","binaryIUObjectCookie","mfcc10_kurtosis","keywordIUObjectSink","NumInflectedVerbs","mfcc7_skewness","keywordIUObjectCookie","VP_to_AUX_ADJP","binaryIUActionWaterOverflowing","keywordIUPlaceExterior","NumSubordinateConjunctions","getConcretenessScore","VPProportion","VP_to_AUX_VP","C_T","NumDeterminers","mfcc12_kurtosis","AvgVPTypeLengthNonEmbedded","S","mfcc13_kurtosis","MLS","AvgVPTypeLengthEmbedded","DC_T","C_S","MLT","mfcc3_kurtosis","INTJ","T","avg_cos_dist","getSUBTLWordScores","mfcc11_kurtosis","CN_T","mfcc9_kurtosis","CT_T","mfcc5_skewness","INTJ_to_UH","RatioNoun","ADJP","mfcc1_skewness"]

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


# one-time use function to fix bugs in sql table
def update_sql():
    acoustic = pd.read_sql_table("dbank_acoustic", cnx)
    diag     = pd.read_sql_table("diagnosis", cnx)
    # df_new.to_sql("dbank_lexical_numeric", cnx, if_exists='replace', index=False)


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

    title = ""
    for i, name in enumerate(names_arr):
        max_str = "%s max: %.3f " % (names_arr[i], means[i].max())
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


def get_data(lexical_table_name, diagnosis, features=None):
    # Read from sql
    lexical   = pd.read_sql_table(lexical_table_name, cnx)
    acoustic  = pd.read_sql_table("dbank_acoustic", cnx)
    diag      = pd.read_sql_table("diagnosis", cnx)
    demo      = pd.read_sql_table("demographic_imputed", cnx)
    discourse = pd.read_sql_table("dbank_discourse", cnx)

    # Merge
    fv = pd.merge(lexical, acoustic, on=['interview'])

    # Select diagnosis
    diag = diag[diag['diagnosis'].isin(diagnosis)]
    fv = pd.merge(fv,diag)
    # Impute
    # demo['age'].fillna(demo['age'].mean(), inplace=True)

    # Add demographics
    fv = pd.merge(fv,demo)
    
    # Add discourse
    fv = pd.merge(fv,discourse, on=['interview'])

    # Randomize
    fv = fv.sample(frac=1,random_state=20)

    # Collect Labels 
    labels = [label[:3] for label in fv['interview']]
    # Split 
    y = fv['dementia'].astype('bool')
    # Clean 
    X = fv.drop(['dementia', 'level_0', 'interview', 'diagnosis', 'gender'], 1)

    X = X.apply(pd.to_numeric, errors='ignore')

    if features:
        X = X[features]
    # Return
    import pdb; pdb.set_trace()

    return X, y, labels


def fraser_features(X): 
    fraser = get_top_50()
    return X[fraser]


def evaluate_model(model, X, y, labels, save_features=False):
    
    model_fs = RandomizedLogisticRegression(C=1)

    # Split into folds using labels 
    label_kfold = LabelKFold(labels, n_folds=10)

    folds  = []
    
    # Feature analysis
    feat_scores = []
    for train_index, test_index in label_kfold:
        print "processing fold: %d" % (len(folds) + 1)

        # Split
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        scores   = []

        for k in XTICKS:
            indices = get_top_pearson_features(X_train, y_train, k)

            # Select k features 
            X_train_fs = X_train[:,indices]
            X_test_fs  = X_test[:,indices]

            model = model.fit(X_train_fs, y_train)
            # summarize the selection of the attributes
            yhat  = model.predict(X_test_fs)                # Predict
            scores.append(accuracy_score(y_test, yhat))     # Save
        
        # ----- save row ----- 
        folds.append(scores)
        # ----- save row ----- 

        # ----- save features ----- 
        if save_features:
            model_fs = model_fs.fit(X_train, y_train) 
            feat_scores.append(model_fs.scores_)
        # -------------------- 

    if save_features:
        feat_scores = np.asarray(feat_scores)  # convert to np array
        feat_scores = feat_scores.mean(axis=0) # squash

        # This command maps scores to features and sorts by score, with the feature name in the first position
        feat_scores = sorted(zip(X.columns, map(lambda x: round(x, 4), model_fs.scores_)), reverse=True,  key=lambda x: x[1])
        feat_scores = pd.DataFrame(feat_scores)

        csv_path = "output/feature_scores/ablation_%d_%d.csv" % (XTICKS.min(), XTICKS.max())
        feat_scores.to_csv(csv_path, index=False)
        print_full(feat_scores)

        
    folds = np.asarray(folds)
    return folds 


def run_experiment(dbname):
    print "Running %s" % dbname
    target = ALZHEIMERS + CONTROL
    source = NON_ALZHEIMERS

    model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)

    Xt, yt, t_labels  = get_data(dbname, target)    
    
    # Xs, ys, s_labels  = get_data(dbname, source)
    # Xdom, ydom, dom_labels = expand_feature_space(Xt, Xs, yt, ys, t_labels, s_labels)

    dbank_lexical = evaluate_model(model, Xt, yt, t_labels,save_features=False)

    plot_results([dbank_lexical], 
                 ["standard"],
                  "# of Features",
                  "Accuracy ")

# ======================================================================================================


if __name__ == '__main__':
    # update_sql()
    run_experiment("dbank_lexical")

    # Xt, yt, t_labels  = get_data("dbank_lexical", target)    
    # Xs, ys, s_labels  = get_data("dbank_lexical", source)
    # Xdom, ydom, dom_labels = expand_feature_space(Xt, Xs, yt, ys, t_labels, s_labels)
    # domain_adaptation()
    # preprocessing_test()