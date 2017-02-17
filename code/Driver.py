import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from FeatureExtractor import feature_sets
from FeatureExtractor import util

# models
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression

# globals
XTICKS = np.arange(1, 100, 1)
REGULARIZATION_CONSTANT = 1


def evaluate_model(model, X, y, labels, save_features=False, group_ablation=False, feature_output_name="features.csv", ticks=XTICKS):

    model_fs = RandomizedLogisticRegression(C=1, random_state=1)
    # Split into folds using labels
    # label_kfold = LabelKFold(labels, n_folds=10)
    group_kfold = GroupKFold(n_splits=10).split(X,y,groups=labels)
    folds  = []
    
    # For feature analysis
    feat_scores = []

    # For ablation study
    # Group ablation study
    feature_groups = feature_sets.get_all_groups()
    ablated = {key: set() for key in feature_groups.keys()}
    roc_ab  = {key: list() for key in feature_groups.keys()}
    roc_ab['true_roc_score'] = []

    for train_index, test_index in group_kfold:
        print "processing fold: %d" % (len(folds) + 1)

        # Split
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        scores   = []

        for k in XTICKS:
            indices = util.get_top_pearson_features(X_train, y_train, k)

            # Select k features
            X_train_fs = X_train[:, indices]
            X_test_fs  = X_test[:, indices]

            model = model.fit(X_train_fs, y_train)
            # summarize the selection of the attributes
            yhat  = model.predict(X_test_fs)                  # Predict
            scores.append(f1_score(y_test, yhat))     # Save
            if group_ablation:
                true_roc_score = roc_auc_score(y_test, yhat)
                roc_ab['true_roc_score'].append(true_roc_score)

                for group in feature_groups.keys():
                    # Get group features
                    features     = feature_groups[group]
                    features_idx = util.get_column_index(features, X)

                    # Get indices
                    indices_ab      = [i for i in indices if i not in features_idx]
                    removed_indices = [i for i in indices if i in features_idx]

                    # Filter
                    X_train_ab = X_train[:, indices_ab]
                    X_test_ab  = X_test[:, indices_ab]

                    # Fit
                    model_ab = model.fit(X_train_ab, y_train)
                    # Predict
                    yhat_ab  = model_ab.predict(X_test_ab)

                    # Save
                    ablated[group].update(X.columns[removed_indices])
                    roc_ab_score = roc_auc_score(y_test, yhat_ab)
                    roc_ab[group].append(roc_ab_score - true_roc_score)

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
        feat_scores = feat_scores.mean(axis=0)  # squash

        # This command maps scores to features and sorts by score, with the feature name in the first position
        feat_scores = sorted(zip(X.columns, map(lambda x: round(x, 4), model_fs.scores_)),
                             reverse=True, key=lambda x: x[1])
        feat_scores = pd.DataFrame(feat_scores)

        csv_path = "output/feature_scores/" + feature_output_name
        feat_scores.to_csv(csv_path, index=False)
        util.print_full(feat_scores)

    if group_ablation:
        roc_ab = pd.DataFrame(roc_ab).mean()
        print "======================="
        print "True AUC Score: %f" % roc_ab['true_roc_score']
        print "=======================\n\n"

        for group in ablated.keys():
            print "-----------------------"
            print "Group: %s " % group
            print "Removed: %s" % list(ablated[group])
            print "Change in AUC: %f" % (roc_ab[group])
            print "-----------------------\n"

    folds = np.asarray(folds)
    return folds
