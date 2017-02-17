import numpy as np
# import pandas as pd
import get_data
from scipy import stats
from FeatureExtractor import util
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
# from sklearn.decomposition import PCA
from FeatureExtractor import feature_sets


ALZHEIMERS     = ['PossibleAD', 'ProbableAD']
CONTROL        = ['Control']
NON_ALZHEIMERS = ['MCI', 'Memory', 'Other', 'Vascular']


class DementiaCV(object):
    """Abstract Base Class for Dementia Cross Validation
        Will be 
    """

    def __init__(self, model, metric):
        super(DementiaCV, self).__init__()
        self.model = model
        self.metric = metric

        self.adapt_methods = None
        self.method_range = {}

        # Results
        self.results    = {}
        self.best_score = {}
        self.best_k     = {}

    def train_all(self):
        for method in self.adapt_methods:
            self.train_model(method)

    def split_data(self, adapt_method):
        raise NotImplementedError()

    def train_model(self, adapt_method, k_range=None):
        if adapt_method not in self.adapt_methods:
            raise KeyError('Model not one of: %s' % self.models)

        output = []
        print "\nTraining %s" % adapt_method
        print "==========================="
        for idx, fold in enumerate(self.split_data(adapt_method)):
            print "Processing fold: %i" % idx
            X_train, y_train = fold["X_train"], fold["y_train"].ravel()  # Ravel flattens a (n,1) array into (n, )
            X_test, y_test   = fold["X_test"], fold["y_test"].ravel()
            scores = []
            nfeats = X_train.shape[1]
            feats = util.get_top_pearson_features(X_train, y_train, nfeats)
            if not k_range:
                k_range = range(1, nfeats)
            for k in k_range:
                indices = feats[:k]
                # Select k features
                X_train_fs = X_train[:, indices]
                X_test_fs  = X_test[:, indices]

                model = self.model.fit(X_train_fs, y_train)
                # summarize the selection of the attributes
                yhat  = model.predict(X_test_fs)         # Predict
                scores.append(self.metric(y_test, yhat))     # Save

            # ----- save row -----
            output.append(scores)

        self.results[adapt_method]    = np.asarray(output)
        self.best_k[adapt_method]     = np.array(k_range)[np.argmax(np.mean(output, axis=0))]
        self.best_score[adapt_method] = np.max(np.mean(output, axis=0))


class DomainAdapter(DementiaCV):
    """docstring for DomainAdapter"""

    def __init__(self, model, metric, with_disc=True):
        super(DomainAdapter, self).__init__(model, metric)
        Xt, yt, lt, Xs, ys, ls = get_data.get_target_source_data(with_disc)
        self.Xt, self.yt, self.lt = Xt.values, yt.values, lt  # Save target data + labels
        self.Xs, self.ys, self.ls = Xs.values, ys.values, ls  # Save source data + labels
        self.adapt_methods = ['target_only', 'source_only', 'relabeled', 'augment', 'coral']

    def split_data(self, adapt_method):
        data = []
        group_kfold = GroupKFold(n_splits=10).split(self.Xt, self.yt, groups=self.lt)
        for train_index, test_index in group_kfold:
            if adapt_method == "target_only":
                X_train = self.Xt[train_index]
                y_train = self.yt[train_index]
                X_test  = self.Xt[test_index]
                y_test  = self.yt[test_index]
                train_labels = np.array(self.lt)[train_index]
            elif adapt_method == 'source_only':
                X_train = self.Xs
                y_train = self.ys
                X_test  = self.Xt[test_index]
                y_test  = self.yt[test_index]
                train_labels = self.ls
            elif adapt_method == 'relabeled':
                # merge'em
                X_merged_relab = np.concatenate([self.Xs, self.Xt[train_index]])
                y_merged_relab = np.concatenate([self.ys, self.yt[train_index]])
                train_labels   = np.concatenate([np.array(self.ls), np.array(self.lt)[train_index]])
                # shuffle
                X_train_relab, y_train_relab, train_labels = shuffle(
                    X_merged_relab, y_merged_relab, train_labels, random_state=1)
                X_train = X_train_relab
                y_train = y_train_relab
                X_test  = self.Xt[test_index]
                y_test  = self.yt[test_index]
            elif adapt_method == 'augment':
                # Extend feature space (train)
                X_merged_aug = self.merge_and_extend_feature_space(self.Xt[train_index], self.Xs)
                y_merged_aug = np.concatenate([self.yt[train_index], self.ys])
                train_labels  = np.concatenate([np.array(self.lt)[train_index], np.array(self.ls)])
                # Extend feature space (test)
                X_test_aug = self.merge_and_extend_feature_space(self.Xt[test_index])
                X_train = X_merged_aug
                y_train = y_merged_aug
                X_test  = X_test_aug
                y_test  = self.yt[test_index]
            elif adapt_method == 'coral':
                # ---------coral------------
                X_train = self.CORAL(self.Xs, self.Xt[train_index])
                y_train = self.ys
                X_test  = self.Xt[test_index]
                y_test  = self.yt[test_index]
                train_labels = self.ls
            else:
                raise KeyError('adapt_method not one of: %s' % self.models)
            fold = {}
            fold["X_train"] = X_train
            fold["y_train"] = y_train
            fold["X_test"]  = X_test
            fold["y_test"]  = y_test
            fold["train_labels"] = train_labels
            data.append(fold)
        return data

    def train_all(self):
        super(DomainAdapter, self).train_all()
        self.train_majority_class()

    def train_majority_class(self):
        output = []
        print "\nTraining Majority Class"
        print "==========================="
        group_kfold = GroupKFold(n_splits=10).split(self.Xt, self.yt, groups=self.lt)
        scores   = []
        for train_index, test_index in group_kfold:
            # Data is same as target_only data
            y_train, y_test = self.yt[train_index], self.yt[test_index]
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

    # map row to new feature space
    # in accordance with 'frustratingly simple' paper
    def merge_and_extend_feature_space(self, X_target, X_source=None):
        X_target_extended = np.concatenate([X_target, np.zeros(X_target.shape), X_target], axis=1)
        if X_source is None:
            return X_target_extended
        else:
            X_source_extended = np.concatenate([X_source, X_source, np.zeros(X_source.shape)], axis=1)
            return np.concatenate([X_target_extended, X_source_extended])

        # Following CORAL paper -http://arxiv.org/abs/1511.05547
    # Algorithm 1
    def CORAL(self, Ds, Dt):
        EPS = 1
        # # Normalize
        Ds = (Ds - Ds.mean(axis=0)) / Ds.std(axis=0)
        Dt = (Dt - Dt.mean(axis=0)) / Dt.std(axis=0)

        # #Fill nan
        Ds = np.nan_to_num(Ds)
        Dt = np.nan_to_num(Dt)

        Cs = np.cov(Ds, rowvar=False) + EPS * np.eye(Ds.shape[1])
        Ct = np.cov(Dt, rowvar=False) + EPS * np.eye(Dt.shape[1])
        Ws = util.msqrt(np.linalg.inv(Cs))

        # # assert Ws*Ws == inv(Cs)
        np.testing.assert_array_almost_equal(Ws.dot(Ws), np.linalg.inv(Cs))
        Ds = np.dot(Ds, Ws)              # Whiten
        Ds = np.dot(Ds, util.msqrt(Ct))  # Recolor

        assert not np.isnan(Ds).any()
        return Ds

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!NOTE! HalvesAdapter NEEDS TO BE TESTED!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class HalvesAdapter(DementiaCV):
    """docstring for DomainAdapter"""

    def __init__(self, model, ticks, metric, with_disc=True):
        super(DomainAdapter, self).__init__(model, ticks, metric)
        self.adapt_methods = ['baseline', 'halves', 'strips', 'quadrants']

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

        group_kfold = GroupKFold(n_splits=10).split(X, y, groups=labels)
        data = []
        for train_index, test_index in group_kfold:
            fold = {}
            fold["X_train"] = X.values[train_index]
            fold["y_train"] = y.values[train_index]
            fold["X_test"]  = X.values[test_index]
            fold["y_test"]  = y.values[test_index]
            data.append(fold)
        return data
