import numpy as np
import pandas as pd
import get_data
from scipy import stats
from FeatureExtractor import util
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from FeatureExtractor import feature_sets


ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ['Control']
NON_ALZHEIMERS = ["MCI", "Memory", "Other", "Vascular"]

class DementiaCV(object):
    """Abstract Base Class for Dementia Cross Validation
        Will be 
    """
    def __init__(self,  model, ticks, metric):
        super(DementiaCV, self).__init__()
        self.model = model
        self.ticks = ticks
        self.metric = metric
        
        self.adapt_methods = None

        # Results
        self.results = {}

    def train_all(self):
        for method in self.adapt_methods:
            self.train_model(method)    
    
    def split_data(self, adapt_method):
        raise NotImplementedError()    

    def train_model(self, adapt_method):
        if adapt_method not in self.adapt_methods:
            raise KeyError('Model not one of: %s' % self.models)
        
        output = []
        print "\nTraining %s" % adapt_method
        print "==========================="
        for idx, fold in enumerate(self.split_data(adapt_method)):
            print "Processing fold: %i" % idx
            X_train, y_train = fold["X_train"], fold["y_train"]
            X_test, y_test   = fold["X_test"], fold["y_test"]

            scores = []
            for k in self.ticks:
                indices = util.get_top_pearson_features(X_train, y_train, k)

                # Select k features
                X_train_fs = X_train[:, indices]
                X_test_fs  = X_test[:, indices]

                model = self.model.fit(X_train_fs, y_train)
                # summarize the selection of the attributes
                yhat  = model.predict(X_test_fs)         # Predict
                scores.append(self.metric(y_test, yhat))     # Save

            # ----- save row -----
            output.append(scores)

        self.results[adapt_method] = np.asarray(output)

        
class DomainAdapter(DementiaCV):
    """docstring for DomainAdapter"""
    def __init__(self, model, ticks, metric, with_disc=True):
        super(DomainAdapter, self).__init__(model, ticks, metric)
        Xt, yt, lt, Xs, ys, ls = get_data.get_target_source_data(with_disc)
        self.Xt, self.yt, self.lt = Xt.values, yt.values, lt  # Save target data + labels
        self.Xs, self.ys, self.ls = Xs.values, ys.values, ls  # Save source data + labels
        
        self.group_kfold = GroupKFold(n_splits=10).split(Xt,yt,groups=lt)
        self.adapt_methods = ['target_only','source_only','relabeled','augment','coral']
        
    def split_data(self, adapt_method):
        data = []
        for train_index, test_index in self.group_kfold:
            if adapt_method == "target_only":
                X_train = self.Xt[train_index]
                y_train = self.yt[train_index]
                X_test  = self.Xt[test_index]
                y_test  = self.yt[test_index]
            elif adapt_method == 'source_only':
                X_train = self.Xs
                y_train = self.ys
                X_test  = self.Xt[test_index]
                y_test  = self.yt[test_index]
            elif adapt_method == "relabeled":
                # merge'em
                X_merged_relab = np.concatenate([self.Xs, self.Xt[train_index]])
                y_merged_relab = np.concatenate([self.ys, self.yt[train_index]])
                # shuffle 
                X_train_relab, y_train_relab = shuffle(X_merged_relab, y_merged_relab, random_state=1)
                X_train = X_train_relab
                y_train = y_train_relab 
                X_test  = self.Xt[test_index]
                y_test  = self.yt[test_index]
            elif adapt_method == 'augment':
                # Extend feature space (train)
                X_merged_aug = merge_and_extend_feature_space(self.Xt[train_index], self.Xs)
                y_merged_aug = np.concatenate([self.yt[train_index], self.ys])
                # Extend feature space (test)
                X_test_aug = merge_and_extend_feature_space(self.Xt[test_index])
                X_train = X_merged_aug
                y_train = y_merged_aug
                X_test  = X_test_aug
                y_test  = self.yt[test_index]
            elif adapt_method == 'coral':
                # ---------coral------------
                Xs_train_coral = CORAL(self.Xs, self.Xt[train_index])
                X_train = Xs_train_coral
                y_train = self.ys
                X_test  = self.Xt[test_index]
                y_test  = self.yt[test_index]
            else:
                raise KeyError('adapt_method not one of: %s' % self.models)
            
            fold = {}
            fold["X_train"] = X_train
            fold["y_train"] = y_train
            fold["X_test"] = X_test
            fold["y_test"] = y_test
            data.append(fold)
        return data

    def train_all(self):
        super(DomainAdapter, self).train_all()
        self.train_majority_class()        

    def train_majority_class(self):
        output = []
        print "\nTraining Majority Class"
        print "==========================="
        for train_index, test_index in self.group_kfold:
            # Data is same as target_only data
            y_train, y_test = self.yt.values[train_index], self.yt.values[test_index]
            scores   = []
            labels = np.array(self.lt)[train_index]
            patient_ids = np.unique(labels)
            maj = []
            for patient in patient_ids:
                ind = np.where(labels == patient)[0]
                maj.append(stats.mode(y_train[ind])[0][0][0])
            maj = stats.mode(maj)[0]

            yhat = np.full(y_test.shape, maj, dtype=bool)
            scores.append(self.metric(y_test, yhat))     # Save
        # ----- save row -----
        output.append(scores)
        # ----- save row -----
        self.results['majority_class'] = np.asarray(output)


class HalvesAdapter(DementiaCV):
    """docstring for DomainAdapter"""
    def __init__(self, model, ticks, metric, with_disc=True):
        super(DomainAdapter, self).__init__(model, ticks, metric)
        self.adapt_methods = ['baseline','halves','strips','quadrants']
        
    def split_data(self, adapt_method):
        to_exclude = feature_sets.get_general_keyword_features()

        if adapt_method == "baseline":
            X, y, labels = get_data(exclude_features=to_exclude, with_spatial=False)
        elif adapt_method == "halves":
            X, y, labels = get_data(spacial_db="dbank_spatial_halves", exclude_features=to_exclude)
        elif adapt_method == "strips":
            X, y, labels = get_data(spacial_db="dbank_spatial_strips", exclude_features=to_exclude)
        elif adapt_method == "quadrants":
            X, y, labels = get_data(spacial_db="dbank_spatial_quadrants", exclude_features=to_exclude)
        else:
            raise KeyError('adapt_method not one of: %s' % self.models)
            
        group_kfold = GroupKFold(n_splits=10).split(X,y,groups=labels)
        data = []
        for train_index, test_index in group_kfold:
            fold = {}
            fold["X_train"] = X.values[train_index]
            fold["y_train"] = y.values[train_index]
            fold["X_test"] = X.values[test_index]
            fold["y_test"] = y.values[test_index]
            data.append(fold)
        return data


# Helper function to map row to new feature space
# in accordance with 'frustratingly simple' paper
def merge_and_extend_feature_space(X_target, X_source=None):
    X_target_extended = np.concatenate([X_target, np.zeros(X_target.shape), X_target], axis=1)
    if X_source is None:
        return X_target_extended
    else:
        X_source_extended = np.concatenate([X_source, X_source, np.zeros(X_source.shape)], axis=1)
        return np.concatenate([X_target_extended, X_source_extended])
     

# # Given a X, y, return the optimal number of features 
# def nested_cv_feature_selection(model, X, y, labels, stepsize=5):
#     group_kfold = GroupKFold(n_splits=3).split(X,y,groups=labels)
#     folds  = []
#     nfeat = np.arange(40, X.shape[1], stepsize)
#     import pdb; pdb.set_trace()
#     for train_index, test_index in group_kfold:
#         print "processing fold: %d" % (len(folds) + 1)
#         # Split
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         scores = []
#         feat_corr = util.get_top_pearson_features(X_train, y_train, nfeat[-1])
#         for k in nfeat:
#             indices = feat_corr[:k]
#             # Select k features
#             X_train_fs = X_train[:, indices]
#             X_test_fs  = X_test[:, indices]

#             model = model.fit(X_train_fs, y_train)          # Train     
#             yhat  = model.predict(X_test_fs)                # Predict
#             scores.append(accuracy_score(y_test, yhat))     # Save

#         # ----- save row -----
#         folds.append(scores)
#         # ----- save row -----
        
#     folds = np.asarray(folds)
#     max_ind = np.argmax(np.mean(folds, axis=0))
#     import pdb; pdb.set_trace()
#     return nfeat[max_ind]


# This baseline splits the target data into 10 folds, and evaluates on a source-trained model for each fold
# Selects features by looking at correlations between feature/label in the source data
# def run_baseline_two(model, Xt, yt, lt, Xs, ys, ls, ticks):
#     # Split into folds using labels
#     group_kfold = GroupKFold(n_splits=10).split(Xt,yt,groups=lt)
#     scores   = []
#     nfeats   = []
#     for train_index, test_index in group_kfold:
#         # Split
#         # Here we train on the entire source data (hence no train index to Xs or ys)
#         X_train, X_test = Xs.values, Xt.values[test_index]
#         y_train, y_test = ys.values, yt.values[test_index]

#         opt_features = nested_cv_feature_selection(model, X_train, y_train, ls)
#         indices = util.get_top_pearson_features(X_train, y_train, opt_features)
#         nfeats.append(opt_features)
#         import pdb; pdb.set_trace()
#         X_train_fs = X_train[:, indices]
#         X_test_fs  = X_test[:, indices]

#         model = model.fit(X_train_fs, y_train)
#         # summarize the selection of the attributes
#         yhat  = model.predict(X_test_fs)                # Predict
#         scores.append(accuracy_score(y_test, yhat))     # Save
        
#     return (scores, nfeats)

# This baseline splits the target data into 10 folds, and evaluates on a source-trained model for each fold
# Selects features by looking at correlations between feature/label in the source data
def run_baseline_majority_class(Xt, yt, lt, ticks):
    # Split into folds using labels
    group_kfold = GroupKFold(n_splits=10).split(Xt,yt,groups=lt)
    folds  = []

    for train_index, test_index in group_kfold:
        print "processing fold: %d" % (len(folds) + 1)

        # Split
        # Here we train on the entire source data (hence no train index to Xs or ys)
        X_train, X_test = Xt.values[train_index], Xt.values[test_index]
        y_train, y_test = yt.values[train_index], yt.values[test_index]

        scores   = []
        labels = np.array(lt)[train_index]
        patient_ids = np.unique(labels)
        maj = []
        for patient in patient_ids:
            ind = np.where(labels == patient)[0]
            maj.append(stats.mode(y_train[ind])[0][0][0])
        maj = stats.mode(maj)[0]

        yhat = np.full(y_test.shape, maj, dtype=bool)
        scores.append(f1_score(y_test, yhat))     # Save
        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
    folds = np.asarray(folds)
    return folds

# This baseline splits the target data into 10 folds, and evaluates on a source-trained model for each fold
# Selects features by looking at correlations between feature/label in the source data
def run_baseline_two(model, Xt, yt, lt, Xs, ys, ls, ticks):
    # Split into folds using labels
    group_kfold = GroupKFold(n_splits=10).split(Xt,yt,groups=lt)
    folds  = []

    for train_index, test_index in group_kfold:
        print "processing fold: %d" % (len(folds) + 1)

        # Split
        # Here we train on the entire source data (hence no train index to Xs or ys)
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
            scores.append(f1_score(y_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
        
    folds = np.asarray(folds)
    return folds


# This baseline adds the entire source data to each training fold and calls it target
def run_baseline_four(model, Xt, yt, lt, Xs, ys, ls, ticks):
    # Split into folds using labels

    group_kfold = GroupKFold(n_splits=10).split(Xt,yt,groups=lt)
    folds  = []

    for train_index, test_index in group_kfold:
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
            scores.append(f1_score(yt_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
        
    folds = np.asarray(folds)
    return folds


# This baseline adds the entire source data to each training fold and calls it target
def run_frustratingly_simple(model, Xt, yt, lt, Xs, ys, ls, ticks):
    # Split into folds using labels

    group_kfold = GroupKFold(n_splits=10).split(Xt,yt,groups=lt)
    folds  = []

    for train_index, test_index in group_kfold:
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
            scores.append(f1_score(yt_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
        
    folds = np.asarray(folds)
    return folds


def get_top_correlated_features_frust(Xt, yt, Xs, ys):
    Xs_train, Xt_train = Xs.values, Xt.values
    ys_train, yt_train = ys.values, yt.values
    X_merged = merge_and_extend_feature_space(Xt_train, Xs_train)
    y_merged = np.concatenate([yt_train, ys_train])
    indices  = util.get_top_pearson_features(X_merged, y_merged, 1000, return_correlation=True)
    for i in indices[1:].index:
        if i < 353:
            print indices.loc[i], Xs.columns[i] + "_both"
        elif i < 706:
            print indices.loc[i], Xs.columns[i - 353] + "_source_only"
        else:
            print indices.loc[i], Xs.columns[i - 706] + "_target_only"

def runPCA(Xt, Xs):
    # Split into folds using labels
    Xs_coral = CORAL(Xs, Xt)
    pca = PCA(n_components=2)
    pca.fit(Xs_coral)
    Xs_pca = pca.transform(Xs_coral)

    dim = Xs_coral.shape
    ones = np.ones(Xs_coral.shape[0])
    Xs_centered = Xs_coral - np.outer(ones, pca.mean_)
    proj = np.dot(Xs_centered, pca.components_[:2].T)



def get_top_correlated_features(Xs, ys):
    Xs_train = Xs.values
    ys_train = ys.values
    indices  = util.get_top_pearson_features(Xs_train, ys_train, 177, return_correlation=True)
    for i in indices[1:].index:
        print indices.loc[i], Xs.columns[i]


# This baseline adds the entire source data to each training fold and calls it target
def run_coral(model, Xt, yt, lt, Xs, ys, ls, ticks, metric):
    # Split into folds using labels
    group_kfold = GroupKFold(n_splits=10).split(Xt,yt,groups=lt)
    folds  = []

    for train_index, test_index in group_kfold:
        print "processing fold: %d" % (len(folds) + 1)
        # Split
        # Here we train on the entire source data (hence no index to Xs or ys)
        Xs_train, Xt_train, Xt_test = Xs.values, Xt.values[train_index], Xt.values[test_index]
        ys_train, yt_train, yt_test = ys.values, yt.values[train_index], yt.values[test_index]
        
        # Realign (training)
        Xs_train = CORAL(Xs_train, Xt_train)
    
        # # Normalize test 
        # Xt_test = (Xt_test - Xt_test.mean(axis=0)) / Xt_test.std(axis=0)
        # Xt_test = np.nan_to_num(Xt_test)

        scores   = []

        for k in ticks:
            indices = util.get_top_pearson_features(Xs_train, ys_train, k)

            # Select k features
            X_train_fs = Xs_train[:, indices]
            X_test_fs  = Xt_test[:, indices]

            model = model.fit(X_train_fs, ys_train)

            # summarize the selection of the attributes
            yhat  = model.predict(X_test_fs)         # Predict
            scores.append(metric(yt_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)
        # ----- save row -----
    folds = np.asarray(folds)
    return folds


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
def CORAL(Ds, Dt):
    EPS = 1
    # # Normalize
    Ds = (Ds - Ds.mean(axis=0)) / Ds.std(axis=0)
    Dt = (Dt - Dt.mean(axis=0)) / Dt.std(axis=0)
    
    # #Fill nan 
    Ds = np.nan_to_num(Ds)
    Dt = np.nan_to_num(Dt)

    Cs = np.cov(Ds,rowvar=False) + EPS * np.eye(Ds.shape[1])
    Ct = np.cov(Dt,rowvar=False) + EPS * np.eye(Dt.shape[1])
    Ws = util.msqrt(np.linalg.inv(Cs))
    
    # # assert Ws*Ws == inv(Cs)
    np.testing.assert_array_almost_equal(Ws.dot(Ws), np.linalg.inv(Cs))
    Ds = np.dot(Ds,Ws)              # Whiten
    Ds = np.dot(Ds,util.msqrt(Ct))  # Recolor

    assert not np.isnan(Ds).any()
    return Ds