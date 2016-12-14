import numpy as np
import pandas as pd
import util
from sklearn.utils import shuffle
from sklearn.cross_validation import LabelKFold
from sklearn.metrics import accuracy_score

ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ['Control']
NON_ALZHEIMERS = ["MCI", "Memory", "Other", "Vascular"]


# Helper function to map row to new feature space
# in accordance with 'frustratingly simple' paper
def merge_and_extend_feature_space(X_target, X_source=None):
    X_target_extended = np.concatenate([X_target, np.zeros(X_target.shape), X_target], axis=1)
    if X_source is None:
        return X_target_extended
    else:
        X_source_extended = np.concatenate([X_source, X_source, np.zeros(X_source.shape)], axis=1)
        return np.concatenate([X_target_extended, X_source_extended])
     
# This baseline splits the target data into 10 folds, and evaluates on a source-trained model for each fold
# Selects features by looking at correlations between feature/label in the source data
def run_baseline_two(model, Xt, yt, lt, Xs, ys, ls, ticks):
    # Split into folds using labels
    label_kfold = LabelKFold(lt, n_folds=10)
    folds  = []

    for train_index, test_index in label_kfold:
        print "processing fold: %d" % (len(folds) + 1)

        # Split
        # Here we train on the entire source data (hence no index to Xs or ys)
        X_train, X_test = Xs.values, Xt.values[test_index]
        y_train, y_test = ys.values, yt.values[test_index]

        scores   = []

        for k in ticks:
            indices = util.get_top_pearson_features(X_train, y_train, k)

            # Select k features
            X_train_fs = X_train[:, indices]
            X_test_fs  = X_test[:, indices]

            model = model.fit(X_train_fs, y_train)

            # summarize the selection of the attributes
            yhat  = model.predict(X_test_fs)         # Predict
            scores.append(accuracy_score(y_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
        
    folds = np.asarray(folds)
    return folds


# This baseline splits the target data into 10 folds, and evaluates on a source-trained model for each fold
# Differs from bl 2 in the way that it chooses features 
# bl3 selects features by looking at correlations between feature/label in the target data (but then uses source data's features )
def run_baseline_three(model, Xt, yt, lt, Xs, ys, ls, ticks):
    # Split into folds using labels
    label_kfold = LabelKFold(lt, n_folds=10)
    folds  = []

    for train_index, test_index in label_kfold:
        print "processing fold: %d" % (len(folds) + 1)

        # Split
        # Here we train on the entire source data (hence no index to Xs or ys)
        X_train, X_test = Xs.values, Xt.values[test_index]
        y_train, y_test = ys.values, yt.values[test_index]

        Xt_train = Xt.values[train_index]
        yt_train = yt.values[train_index]

        scores   = []

        for k in ticks:
            indices = util.get_top_pearson_features(Xt_train, yt_train, k)

            # Select k features
            X_train_fs = X_train[:, indices]
            X_test_fs  = X_test[:, indices]

            model = model.fit(X_train_fs, y_train)

            # summarize the selection of the attributes
            yhat  = model.predict(X_test_fs)         # Predict
            scores.append(accuracy_score(y_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
        
    folds = np.asarray(folds)
    return folds


# This baseline adds the entire source data to each training fold and calls it target
def run_baseline_four(model, Xt, yt, lt, Xs, ys, ls, ticks):
    # Split into folds using labels

    label_kfold = LabelKFold(lt, n_folds=10)
    folds  = []

    for train_index, test_index in label_kfold:
        print "processing fold: %d" % (len(folds) + 1)
        # Split
        # Here we train on the entire source data (hence no index to Xs or ys)
        Xs_train, Xt_train, Xt_test = Xs.values, Xt.values[train_index], Xt.values[test_index]
        ys_train, yt_train, yt_test = ys.values, yt.values[train_index], yt.values[test_index]
           
        # merge'em
        X_merged = np.concatenate([Xs_train, Xt_train])
        y_merged = np.concatenate([ys_train, yt_train])

        # shuffle 
        X_train, y_train = shuffle(X_merged, y_merged, random_state=1)

        scores   = []

        for k in ticks:
            indices = util.get_top_pearson_features(X_train, y_train, k)

            # Select k features
            X_train_fs = X_train[:, indices]
            X_test_fs  = Xt_test[:, indices]
            model = model.fit(X_train_fs, y_train)

            # summarize the selection of the attributes
            yhat  = model.predict(X_test_fs)         # Predict
            scores.append(accuracy_score(yt_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
        
    folds = np.asarray(folds)
    return folds


# This baseline adds the entire source data to each training fold and calls it target
def run_frustratingly_simple(model, Xt, yt, lt, Xs, ys, ls, ticks):
    # Split into folds using labels

    label_kfold = LabelKFold(lt, n_folds=10)
    folds  = []

    for train_index, test_index in label_kfold:
        print "processing fold: %d" % (len(folds) + 1)
        # Split
        # Here we train on the entire source data (hence no index to Xs or ys)
        Xs_train, Xt_train, Xt_test = Xs.values, Xt.values[train_index], Xt.values[test_index]
        ys_train, yt_train, yt_test = ys.values, yt.values[train_index], yt.values[test_index]
        
        # Extend feature space (training)
        X_merged = merge_and_extend_feature_space(Xt_train, Xs_train)
        y_merged = np.concatenate([yt_train, ys_train])

        # Extend feature space (testing)
        X_test_merged = merge_and_extend_feature_space(Xt_test)

        scores   = []

        for k in ticks:
            indices = util.get_top_pearson_features(X_merged, y_merged, k)

            # Select k features
            X_train_fs = X_merged[:, indices]
            X_test_fs  = X_test_merged[:, indices]

            model = model.fit(X_train_fs, y_merged)

            # summarize the selection of the attributes
            yhat  = model.predict(X_test_fs)         # Predict
            scores.append(accuracy_score(yt_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
        
    folds = np.asarray(folds)
    return folds


def concat_and_shuffle(X1, y1, l1, X2, y2, l2, random_state=1):
    # Coerce all arguments to dataframes
    X1, X2 = pd.DataFrame(X1), pd.DataFrame(X2) 
    y1, y2 = pd.DataFrame(y1), pd.DataFrame(y2) 
    l1, l2 = pd.DataFrame(l1), pd.DataFrame(l2) 

    X_concat = X1.append(X2, ignore_index=True)
    y_concat = y1.append(y2, ignore_index=True)
    l_concat = l1.append(l2, ignore_index=True)
    return shuffle(X_concat, y_concat, l_concat, random_state=random_state)


# Following "frustratingly simple"
def expand_feature_space(Xtarget, Xsource, Ytarget, Ysource, t_labels, s_labels):
    tgt = [s + "_tgt" for s in Xtarget.columns]
    src = [s + "_src" for s in Xsource.columns]
    shr = [s + "_shr" for s in Xsource.columns]
    
    # Strangly, results improve when you add a 2nd and 3rd shared feature space 
    # shr2 = [s + "_shr_two" for s in Xsource.columns]
    # shr3 = [s + "_shr_three" for s in Xsource.columns]

    new_features = shr + src + tgt 

    data = []

    for index, row in Xtarget.iterrows():
        new_row = triplicate_row(row,target=True)
        data.append(new_row)

    for index, row in Xsource.iterrows():
        new_row = triplicate_row(row,target=False)
        data.append(new_row)

    df = pd.DataFrame(data, columns=new_features)
    X, y, labels = df, Ytarget.append(Ysource), t_labels + s_labels

    return shuffle(X, y, labels, random_state=1)


# Following CORAL paper -http://arxiv.org/abs/1511.05547
# Algorithm 1
def CORAL(Dt, Ds, Ytarget, Ysource, t_labels, s_labels):
    Cs = Ds.cov().values + np.eye(Ds.shape[1])
    Ct = Dt.cov().values + np.eye(Dt.shape[1])
    print "test"
    print "test"
