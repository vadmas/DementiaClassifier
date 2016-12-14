from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.cross_validation import LabelKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from FeatureExtractor import feature_sets
from FeatureExtractor import util
from FeatureExtractor import domain_adaptation

# models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
import itertools

# ====================
import warnings
warnings.filterwarnings('ignore')

# ====================
# globals
XTICKS = np.arange(40, 1200, 20)
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


def plot_results(results_arr, names_arr, with_err=True):

    if len(results_arr) != len(names_arr):
        print "error: Results and names array not same length"
        return

    xlabel = "# of Features"
    ylabel = "Accuracy"

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

    if with_err:
        plot = df.plot(yerr=err_dict, title=title)
    else:
        plot = df.plot(title=title)

    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)
    plt.show()


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


def get_data(diagnosis=ALZHEIMERS+CONTROL, include_features=None, exclude_features=None, polynomial=True):
    # Read from sql
    lexical    = pd.read_sql_table("dbank_lexical", cnx)
    acoustic   = pd.read_sql_table("dbank_acoustic", cnx)
    diag       = pd.read_sql_table("diagnosis", cnx)
    demo       = pd.read_sql_table("demographic_imputed", cnx)
    discourse  = pd.read_sql_table("dbank_discourse", cnx)
    infounits  = pd.read_sql_table("dbank_cookie_theft_info_units", cnx)
    spacial    = pd.read_sql_table("dbank_spatial_halves", cnx)

    # Merge lexical and acoustic
    fv = pd.merge(lexical, acoustic, on=['interview'])

    # Add diagnosis
    diag = diag[diag['diagnosis'].isin(diagnosis)]
    fv = pd.merge(fv, diag)

    # Add demographics
    fv = pd.merge(fv, demo)

    # Add discourse
    fv = pd.merge(fv, discourse, on=['interview'])

    # Add infounits
    fv = pd.merge(fv, infounits, on=['interview'])

    # Add spacial
    # (Add Polynomial)
    if polynomial:
        spacial = spacial.drop(exclude_features, errors='ignore')
        spacial = make_polynomial_terms(spacial)
    fv = pd.merge(fv, spacial, on=['interview'])

    # Randomize
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


def evaluate_model(model, X, y, labels):

    # Split into folds using labels
    label_kfold = LabelKFold(labels, n_folds=10)
    folds  = []

    for train_index, test_index in label_kfold:
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

            model = model.fit(X_train_fs, y_train)          # Fit   
            yhat  = model.predict(X_test_fs)                # Predict
            scores.append(accuracy_score(y_test, yhat))     # Save

        # ----- save row -----
        folds.append(scores)

    folds = np.asarray(folds)
    return folds


def run_experiment():
    model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)

    X_halves, y_halves, labels_halves = get_data(spacial_db="dbank_spatial_halves",)
    halves_model = evaluate_model(model, X_halves, y_halves, labels_halves)

    X_strips, y_strips, labels_strips = get_data(spacial_db="dbank_spatial_strips", exclude_features=to_exclude)
    strips_model = evaluate_model(model, X_strips, y_strips, labels_strips)

    X_quadrants, y_quadrants, labels_quadrants = get_data(
        spacial_db="dbank_spatial_quadrants", exclude_features=to_exclude)
    quadrants_model = evaluate_model(model, X_quadrants, y_quadrants, labels_quadrants)

    plot_results([halves_model, strips_model, quadrants_model], [
                 "halves_model", "strips_model", "quadrants_model"], with_err=False)


# ======================================================================================================
if __name__ == '__main__':
    run_experiment()
