from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.cross_validation import LabelKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# models
from sklearn import linear_model
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier


from FeatureExtractor.fraser_feature_set import get_top_50

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

    lexical  = pd.read_sql_table("dbank_lexical", cnx)
    acoustic = pd.read_sql_table("dbank_acoustic", cnx)

    # Merge
    fv = pd.merge(lexical, acoustic, on=['interview', 'dementia'])

    # Randomize
    fv = fv.sample(frac=1,random_state=20)

    # Collect Labels 
    labels = [label[:3] for label in fv['interview']]

    # Split 
    y = fv['dementia'].astype('bool')
    X = fv.drop(['dementia', 'interview', 'level_0'], 1)

    # Return
    return X, y, labels


def fraser_features(X):
    fraser = get_top_50()
    return X[fraser]



def model_comparison():
    X, y, labels = get_data()
    rf = RandomForestClassifier(n_estimators=20)
    svm = SVC()
    l1 = linear_model.LogisticRegression(penalty='l1') 
    l2 = linear_model.LogisticRegression(penalty='l2') 

    # Split into folds using labels 
    label_kfold = LabelKFold(labels, n_folds=10)

    rf_folds  = []
    svm_folds = []
    l1_folds  = []
    l2_folds  = []

    for train_index, test_index in label_kfold:
        print "processing fold: %d" % (len(rf_folds) + 1)
        
        # Split
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        nfeat = X_train.shape[1]

        rf_scores = []
        svm_scores = []
        l1_scores = []
        l2_scores = []

        for k in xrange(1,nfeat):
            # Perform feature selection
            fit = SelectKBest(f_classif, k=k).fit(X_train, y_train)
            indices = fit.get_support(indices=True)

            # Select k features 
            X_train_fs = X_train[:,indices]
            X_test_fs  = X_test[:,indices]

            # Fit    
            rf.fit(X_train_fs, y_train)
            svm.fit(X_train_fs, y_train)
            l1.fit(X_train_fs, y_train)
            l2.fit(X_train_fs, y_train)

            # Predict
            rf_yhat  = rf.predict(X_test_fs)           
            svm_yhat = svm.predict(X_test_fs)           
            l1_yhat  = l1.predict(X_test_fs)           
            l2_yhat  = l2.predict(X_test_fs)           
            # Save

            rf_scores.append(f1_score(y_test, rf_yhat))
            svm_scores.append(f1_score(y_test, svm_yhat))
            l1_scores.append(f1_score(y_test, l1_yhat))
            l2_scores.append(f1_score(y_test, l2_yhat))
        
        # ----- save row ----- 
        rf_folds.append(rf_scores)
        svm_folds.append(svm_scores)
        l1_folds.append(l1_scores)
        l2_folds.append(l2_scores)
        # -------------------- 

    rf_folds  = np.asarray(rf_folds)
    svm_folds = np.asarray(svm_folds)
    l1_folds  = np.asarray(l1_folds)
    l2_folds  = np.asarray(l2_folds)
    
    rf_means  = np.mean(rf_folds, axis=0)
    svm_means = np.mean(svm_folds, axis=0)
    l1_means  = np.mean(l1_folds, axis=0)
    l2_means  = np.mean(l2_folds, axis=0)

    rf_std  = np.std(rf_folds, axis=0)
    svm_std = np.std(svm_folds, axis=0)
    l1_std  = np.std(l1_folds, axis=0)
    l2_std  = np.std(l2_folds, axis=0)

    print "rf  max: %f " % rf_means.max()
    print "svm max: %f " % svm_means.max()
    print "l1  max: %f " % l1_means.max()
    print "l2  max: %f " % l2_means.max()

    title = "Model comparison, rf max: %f, svm max: %f, l1 max: %f, l2 max: %f " % (rf_means.max(), svm_means.max(), l1_means.max(), l2_means.max())

    df = pd.DataFrame({"random_forest":rf_means, "SVM":svm_means, "l1 logistic regression":l1_means, "l2 logistic regression":l2_means})
    plot = df.plot(yerr={"random_forest":rf_std, "SVM":svm_std, "l1 logistic regression":l1_std, "l2 logistic regression":l2_std,}, title=title)
    plot.set_xlabel("# of Features")
    plot.set_ylabel("Accuracy (F1 Score)")
    plt.show()


def stability_selection():
    X, y, labels = get_data()
    rf = RandomForestClassifier(n_estimators=20)
    svm = SVC()
    l1 = linear_model.LogisticRegression(penalty='l1') 
    l2 = linear_model.LogisticRegression(penalty='l2') 

    # Split into folds using labels 
    label_kfold = LabelKFold(labels, n_folds=10)

    rf_folds  = []
    svm_folds = []
    l1_folds  = []
    l2_folds  = []

    for train_index, test_index in label_kfold:
        print "Processing fold: %d" % (len(rf_folds) + 1)
        
        # Split
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        # Perform feature selection
        fit = linear_model.RandomizedLogisticRegression().fit(X_train, y_train)
        indices = fit.get_support(indices=True)

        # Select k features 
        X_train_fs = X_train[:,indices]
        X_test_fs  = X_test[:,indices]

        print "# of features selected: % d" % X_train_fs.shape[1]

        # Fit    
        rf.fit(X_train_fs, y_train)
        svm.fit(X_train_fs, y_train)
        l1.fit(X_train_fs, y_train)
        l2.fit(X_train_fs, y_train)

        # Predict
        rf_yhat  = rf.predict(X_test_fs)           
        svm_yhat = svm.predict(X_test_fs)           
        l1_yhat  = l1.predict(X_test_fs)           
        l2_yhat  = l2.predict(X_test_fs)           
        
        # Save
        rf_folds.append(f1_score(y_test, rf_yhat))
        svm_folds.append(f1_score(y_test, svm_yhat))
        l1_folds.append(f1_score(y_test, l1_yhat))
        l2_folds.append(f1_score(y_test, l2_yhat))
        
        # # ----- save row ----- 
        # rf_folds.append(rf_scores)
        # svm_folds.append(svm_scores)
        # l1_folds.append(l1_scores)
        # l2_folds.append(l2_scores)
        # # -------------------- 

    rf_folds  = np.asarray(rf_folds)
    svm_folds = np.asarray(svm_folds)
    l1_folds  = np.asarray(l1_folds)
    l2_folds  = np.asarray(l2_folds)
    
    rf_means  = np.mean(rf_folds, axis=0)
    svm_means = np.mean(svm_folds, axis=0)
    l1_means  = np.mean(l1_folds, axis=0)
    l2_means  = np.mean(l2_folds, axis=0)

    # rf_std  = np.std(rf_folds, axis=0)
    # svm_std = np.std(svm_folds, axis=0)
    # l1_std  = np.std(l1_folds, axis=0)
    # l2_std  = np.std(l2_folds, axis=0)

    print "rf  max: %f " % rf_means.max()
    print "svm max: %f " % svm_means.max()
    print "l1  max: %f " % l1_means.max()
    print "l2  max: %f " % l2_means.max()

    # title = "Model comparison, rf max: %f, svm max: %f, l1 max: %f, l2 max: %f " % (rf_means.max(), svm_means.max(), l1_means.max(), l2_means.max())

    # df = pd.DataFrame({"random_forest":rf_means, "SVM":svm_means, "l1 logistic regression":l1_means, "l2 logistic regression":l2_means})
    # plot = df.plot(yerr={"random_forest":rf_std, "SVM":svm_std, "l1 logistic regression":l1_std, "l2 logistic regression":l2_std,}, title=title)
    # plot.set_xlabel("# of Features")
    # plot.set_ylabel("Accuracy (F1 Score)")
    # plt.show()



if __name__ == '__main__':
    stability_selection()
