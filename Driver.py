from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.cross_validation import LabelKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from FeatureExtractor import feature_sets
from FeatureExtractor import utils 
# models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression

# ====================
import warnings
warnings.filterwarnings('ignore')

# ====================
# globals
XTICKS = np.arange(50,400,2)
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


# one-time use function to fix bugs in sql table
def update_sql():
    lex_tmp = pd.read_sql_table("dbank_lexical_tmp", cnx)
    cookie  = pd.read_sql_table("dbank_cookie_theft_info_units", cnx)

    # lsrs    = pd.read_sql_table("leftside_rightside_polynomial", cnx)
    # halves  = pd.read_sql_table("dbank_spatial_halves", cnx)

    drop_rsls = ["count_of_leftside_keyword","count_of_rightside_keyword","leftside_keyword_to_non_keyword_ratio","leftside_keyword_type_to_token_ratio","percentage_of_leftside_keywords_mentioned","percentage_of_rightside_keywords_mentioned","rightside_keyword_to_non_keyword_ratio","rightside_keyword_type_to_token_ratio","ratio_left_to_right_keyword_count","ratio_left_to_right_keyword_to_word","ratio_left_to_right_keyword_type_to_token","ratio_left_to_right_keyword_percentage"]
    drop_cookie = cookie.columns.drop('interview')
    lex_tmp = lex_tmp.drop(drop_rsls, axis=1, errors='ignore')
    lex_tmp = lex_tmp.drop(drop_cookie, axis=1, errors='ignore')
    # lex_tmp.to_sql("dbank_lexical", cnx, if_exists='replace', index=False)
    print "fin!"


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


def get_top_pearson_features(X,y,n):
    df = pd.DataFrame(X).apply(pd.to_numeric)
    df['y'] = y
    corr_coeff = df.corr()['y'].abs().sort(inplace=False, ascending=False)
    return corr_coeff.index.values[1:n+1].astype(int)


def make_polynomial_terms(df,keep_linear = True):
    cols = df.columns.drop('interview', errors='ignore')
    df = df.apply(pd.to_numeric, errors='ignore')
    # Squared terms 
    for f in cols:
        df["sqr_" + f] = df[f]**2
    #interaction
    for f1 in cols:
        for f2 in cols:
            if f1 != f2:
                df["intr_" + f1 + "_" + f2] = df[f1]*df[f2]
    if not keep_linear:
        df = df.drop(cols, axis=1, errors='ignore')
    return df

def get_data(spacial_db, diagnosis=ALZHEIMERS+CONTROL, include_features=None, exclude_features=None):
    # Read from sql
    lexical    = pd.read_sql_table("dbank_lexical", cnx)
    acoustic   = pd.read_sql_table("dbank_acoustic", cnx)
    diag       = pd.read_sql_table("diagnosis", cnx)
    demo       = pd.read_sql_table("demographic_imputed", cnx)
    discourse  = pd.read_sql_table("dbank_discourse", cnx)
    infounits  = pd.read_sql_table("dbank_cookie_theft_info_units", cnx)
    spacial     = pd.read_sql_table(spacial_db, cnx)
    
    # Merge lexical and acoustic 
    fv = pd.merge(lexical, acoustic, on=['interview'])

    # Add diagnosis
    diag = diag[diag['diagnosis'].isin(diagnosis)]
    fv = pd.merge(fv,diag)
    
    # Add demographics
    fv = pd.merge(fv,demo)
    
    # Add discourse
    fv = pd.merge(fv,discourse, on=['interview'])

    # Add infounits
    fv = pd.merge(fv,infounits, on=['interview'])

    # Add spacial
    # (Add Polynomial)
    spacial = spacial.drop(exclude_features, errors='ignore')
    spacial = make_polynomial_terms(spacial)
    fv = pd.merge(fv,spacial, on=['interview'])

    # Randomize
    fv = fv.sample(frac=1,random_state=20)

    # Collect Labels 
    labels = [label[:3] for label in fv['interview']]
    # Split 
    y = fv['dementia'].astype('bool')
    
    # Clean 
    drop = ['dementia', 'level_0', 'interview', 'diagnosis', 'gender', 'index','gender_int']
    if exclude_features:
        drop += exclude_features
    X = fv.drop(drop, 1, errors='ignore')

    X = X.apply(pd.to_numeric, errors='ignore')

    if include_features:
        X = X[include_features]

    return X, y, labels


def evaluate_model(model, X, y, labels, save_features=False, group_ablation=False):
    
    model_fs = RandomizedLogisticRegression(C=1, random_state=1)

    # Split into folds using labels 
    label_kfold = LabelKFold(labels, n_folds=10)
    folds  = []
    
    # For feature analysis
    feat_scores = []

    # For ablation study
    # Group ablation study
    feature_groups = feature_sets.get_all_groups()
    ablated = {key:set() for key in feature_groups.keys()}
    roc_ab  = {key:list() for key in feature_groups.keys()}
    roc_ab['true_roc_score'] = []

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

            if group_ablation:            
                true_roc_score = roc_auc_score(y_test, yhat)
                roc_ab['true_roc_score'].append(true_roc_score)

                for group in feature_groups.keys():
                    # Get group features
                    features     = feature_groups[group]
                    features_idx = utils.get_column_index(features,X)
                    
                    # Get indices
                    indices_ab      = [i for i in indices if i not in features_idx]
                    removed_indices = [i for i in indices if i in features_idx]
                    
                    # Filter 
                    X_train_ab = X_train[:,indices_ab]
                    X_test_ab  = X_test[:,indices_ab]

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
        feat_scores = feat_scores.mean(axis=0) # squash

        # This command maps scores to features and sorts by score, with the feature name in the first position
        feat_scores = sorted(zip(X.columns, map(lambda x: round(x, 4), model_fs.scores_)), reverse=True,  key=lambda x: x[1])
        feat_scores = pd.DataFrame(feat_scores)

        csv_path = "output/feature_scores/general_keywords.csv"
        feat_scores.to_csv(csv_path, index=False)
        utils.print_full(feat_scores)

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


def run_experiment():
    model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)
    gen_keyword_features  = feature_sets.get_general_keyword_features()
    switches = ["count_ls_rs_switches"]
    ratio = ["ratio_left_to_right_keyword_count","ratio_left_to_right_keyword_to_word","ratio_left_to_right_keyword_type_to_token","ratio_left_to_right_keyword_percentage"]
    to_exclude = gen_keyword_features + ratio + switches
    
    X_halves, y_halves, labels_halves = get_data(spacial_db="dbank_spatial_halves", exclude_features=to_exclude)
    halves_model = evaluate_model(model, X_halves, y_halves, labels_halves, save_features=False)    
    
    X_strips, y_strips, labels_strips = get_data(spacial_db="dbank_spatial_strips", exclude_features=to_exclude)
    strips_model = evaluate_model(model, X_strips, y_strips, labels_strips, save_features=False)    
    
    X_quadrants, y_quadrants, labels_quadrants   = get_data(spacial_db="dbank_spatial_quadrants", exclude_features=to_exclude)
    quadrants_model = evaluate_model(model, X_quadrants, y_quadrants, labels_quadrants, save_features=False)    

    plot_results([halves_model,strips_model,quadrants_model], ["halves_model","strips_model","quadrants_model"], with_err=False)


# def run_domain_adaptation():
#     model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)

#     target = ALZHEIMERS + CONTROL
#     source = NON_ALZHEIMERS
    
#     info_content_features = feature_sets.get_information_content_features()

#     lsrs_features         = feature_sets.get_leftside_rightside_features()
#     switches = ["count_ls_rs_switches",]

#     X_2, y_2, labels_2  = get_data(target, exclude_features=gen_keyword_features, polynomial=True)

#     print "Running standard_model..."
#     standard_model  = evaluate_model(model, X_0, y_0, labels_0, save_features=True)
#     print "Running lsrs_model..."
#     without_switches = evaluate_model(model, X_1, y_1, labels_1, save_features=True)
#     print "Running lsrs_poly_model..."
#     with_switches    = evaluate_model(model, X_2, y_2, labels_2, save_features=True)

    
#     plot_results([without_switches, with_switches], 
#                  ["Polynomial, small keyword set", "Polynomial, expanded keywords set"],
#                   "# of Features",
#                   "Accuracy",
#                   with_err=False)

#     plot_results([with_switches], 
#                  ["Polynomial, Switch, expanded keywords"],
#                   "# of Features",
#                   "Accuracy",
#                   with_err=False)


# ======================================================================================================
if __name__ == '__main__':
    run_experiment()
    