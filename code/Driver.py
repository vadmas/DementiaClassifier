import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from FeatureExtractor import feature_sets
from FeatureExtractor import util
from FeatureExtractor import domain_adaptation
from scripts.domain_adaptation import DomainAdapter
from scripts import get_data
import seaborn as sns
import scipy.stats as st


# models
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RandomizedLogisticRegression

# ====================
import warnings
warnings.filterwarnings('ignore')

# ====================
# globals
XTICKS = np.arange(1, 300, 1)
REGULARIZATION_CONSTANT = 1


def save_csv(output, results_arr, names_arr, with_err=True, ticks=XTICKS):    
    if len(results_arr) != len(names_arr):
        print "error: Results and names array not same length"
        return

    dfs = []
    for idx, results in enumerate(results_arr):
        arr = np.asarray(results)
        max_ind = np.argmax(np.mean(arr, axis=0))
        df = pd.DataFrame(arr[:,max_ind], columns=['fold_vals'])
        df['model'] = names_arr[idx]
        df['best_k'] = ticks[max_ind]
        dfs.append(df)
    
    dfs = pd.concat(dfs)
    dfs.to_csv(output)
    # ax = sns.barplot(x="model", y="fold_vals", data=dfs, capsize=.2)
    # sns.plt.show()


def save_results(results, name):    
    dim = results.shape
    fmeas = []
    ppv = []
    npv = []
    for fold in range(dim[0]):
        fmeas_fold = []
        ppv_fold = []
        npv_fold = []
        for k in range(dim[1]):
            cf = results[fold][k]
            tn = float(cf[0][0])
            fn = float(cf[1][0])
            tp = float(cf[1][1])
            fp = float(cf[0][1])

            pos_pred_val = tp / (tp + fp)
            neg_pred_val = tn / (tn + fn)
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            fm = (2 * prec * rec) / (prec + rec)

            fmeas_fold.append(fm)
            ppv_fold.append(pos_pred_val)
            npv_fold.append(neg_pred_val)

        fmeas.append(fmeas_fold)
        ppv.append(ppv_fold)
        npv.append(npv_fold)

    fmeas = pd.DataFrame(fmeas)
    ppv   = pd.DataFrame(ppv)
    npv   = pd.DataFrame(npv)

    path = '../docs/output/halves/csvs/' + name
    
    fmeas.to_csv(path + "_fmeas.csv")
    ppv.to_csv(path + "_ppv.csv")
    npv.to_csv(path + "_npv.csv")
    
def plot_results(results_arr, names_arr, with_err=True, ticks=XTICKS):

    if len(results_arr) != len(names_arr):
        print "error: Results and names array not same length"
        return

    xlabel = "# of Features"
    ylabel = "F-Measure"

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
        # title += max_str
        df_dict[name] = means[i]
        err_dict[name] = stdevs[i]

    df = pd.DataFrame(df_dict)
    df.index = ticks

    sns.set_style("whitegrid")

    if with_err:
        plot = df.plot(yerr=err_dict, title=title)
    else:
        plot = df.plot(title=title)

    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)
    plt.show()


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
            scores.append(confusion_matrix(y_test, yhat))     # Save
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


def run_experiment(model):
    name = model.__class__.__name__
    to_exclude = feature_sets.get_general_keyword_features()

    print "Running: %s" % name
    X_baseline, y_baseline, labels_baseline = get_data(exclude_features=to_exclude, with_spatial=False)
    baseline_model = evaluate_model(model, X_baseline, y_baseline, labels_baseline, save_features=False)
    save_results(baseline_model, "baseline" + "_" + name)

    X_halves, y_halves, labels_halves = get_data(spacial_db="dbank_spatial_halves", exclude_features=to_exclude)
    halves_model = evaluate_model(model, X_halves, y_halves, labels_halves, save_features=False)
    save_results(halves_model, "halves" + "_" + name)

    X_strips, y_strips, labels_strips = get_data(spacial_db="dbank_spatial_strips", exclude_features=to_exclude)
    strips_model = evaluate_model(model, X_strips, y_strips, labels_strips, save_features=False)
    save_results(strips_model, "strips" + "_" + name)

    X_quadrants, y_quadrants, labels_quadrants = get_data(spacial_db="dbank_spatial_quadrants", exclude_features=to_exclude)
    quadrants_model = evaluate_model(model, X_quadrants, y_quadrants, labels_quadrants, save_features=False)
    save_results(quadrants_model, "quadrants" + "_" + name)


def get_max_std(path):
    output = pd.read_csv(path,index_col=0)
    means = np.mean(output, axis=0)
    std = np.std(output, axis=0)
    max_ind = np.argmax(means)
    return (means.loc[max_ind], std.loc[max_ind], XTICKS[int(max_ind)])


def print_stats(name):
    path = "../docs/output/halves/csvs/"
    fmeas = pd.read_csv(path + name + "_fmeas.csv",index_col=0)
    npv   = pd.read_csv(path + name + "_npv.csv",index_col=0)
    ppv   = pd.read_csv(path + name + "_ppv.csv",index_col=0)

    avg_fmeas = np.mean(fmeas, axis=0)
    avg_npv = np.mean(npv, axis=0)
    avg_ppv = np.mean(ppv, axis=0)

    max_ind = np.argmax(avg_fmeas)
    print name
    fm_ci = st.t.interval(0.95, len(fmeas[max_ind])-1, loc=np.mean(fmeas[max_ind]), scale=st.sem(fmeas[max_ind]))
    print "F-Measure: %0.3f 95 CI: [%0.3f, %0.3f]" % (avg_fmeas.loc[max_ind], fm_ci[0], fm_ci[1])

    ppv_ci = st.t.interval(0.95, len(ppv[max_ind])-1, loc=np.mean(ppv[max_ind]), scale=st.sem(ppv[max_ind]))
    print "ppv: %0.3f 95 CI: [%0.3f, %0.3f]" % (avg_ppv.loc[max_ind], ppv_ci[0], ppv_ci[1])

    npv_ci = st.t.interval(0.95, len(npv[max_ind])-1, loc=np.mean(npv[max_ind]), scale=st.sem(npv[max_ind]))
    print "npv: %0.3f 95 CI: [%0.3f, %0.3f]" % (avg_npv.loc[max_ind], npv_ci[0], npv_ci[1])


# Baseline:
# 1. No domain adaptation (10fold cv on MCI/Control)
# 2. Train model on Alzheimers, test on 10 folds
#    - Within each fold:
#       - top k correlated feature between X_alz_train & y_alz_train
#       - features selected by correlation between Alzheimers data (NO mci correlation used)
# 3. Train model on Alzheimers, test on 10 folds 
#    - Within each fold:
#       - top k correlated feature between X_mci_train & y_mci_train
#       - Alzhiemers data used, but features selected by correlation between mci data
# 4. Relabel: Train Alzheimers + MCI, test on 10 folds 

# Domain adapt:
# 1. Frustratingly simple 1
# 2. Frustratingly simple 2


def run_domain_adaptation():
    model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)
    da = DomainAdapter(model, XTICKS, f1_score, with_disc=False)
    da.train_model('source_only')
    plot_results([da.results['source_only']], ['source_only'], with_err=True)    
    
    Xt, yt, lt, Xs, ys, ls = get_data.get_target_source_data(with_disc=False)
    bl_2 = domain_adaptation.run_baseline_two(model, Xt, yt, lt, Xs, ys, ls, XTICKS)
    plot_results([bl_2], ['bl_2_without_disc'], with_err=True, ticks=XTICKS)
    


    # da.train_all()
    # da.results
    
    # models = [da.results[method] for method in da.adapt_methods]
    # plot_results(models, da.adapt_methods, with_err=True)    


    # # # Baseline 1
    # print "Running Baseline 0"
    # model_0 = DummyClassifier()
    # bl_0 = domain_adaptation.run_baseline_majority_class(Xt, yt, lt, XTICKS)
    # # plot_results([bl_0], ["baseline_0"], with_err=True)

    # # # Baseline 1
    # print "Running Baseline 1"
    # grid = np.arange(250, 350, 1)
    # bl_1 = evaluate_model(model, Xt, yt, lt, save_features=True, feature_output_name="MCI_without_discourse.csv", ticks=grid)
    # bl_1_disc = evaluate_model(model, Xt_disc, yt_disc, lt_disc, save_features=True, feature_output_name="MCI_without_discourse.csv")
    # save_csv("output/tmp_csvs/target_only.csv",[bl_1_disc, bl_1], ["target_only_with_disc", "target_only_without_disc"])
    
    
    # Baseline 2
    # print "Running Baseline 2"
    # grid = np.arange(170, 210, 1)
    # bl_2 = domain_adaptation.run_baseline_two(model, Xt, yt, lt, Xs, ys, ls, grid)
    # bl_2_disc = domain_adaptation.run_baseline_two(model, Xt_disc, yt_disc, lt_disc, Xs_disc, ys_disc, ls_disc, grid)
    # save_csv("output/tmp_csvs/source_only.csv",[bl_2_disc, bl_2], ["source_only_with_disc", "source_only_without_disc"])
    
    # # # Baseline 4
    # print "Running Baseline 4"
    # grid = np.arange(10, 350, 1)
    # bl_4 = domain_adaptation.run_baseline_four(model, Xt, yt, lt, Xs, ys, ls, grid)
    # bl_4_disc = domain_adaptation.run_baseline_four(model, Xt_disc, yt_disc, lt_disc, Xs_disc, ys_disc, ls_disc, grid)
    # save_csv("output/tmp_csvs/relabeled.csv",[bl_4_disc, bl_4], ["relabelled_with_disc", "relabelled_without_disc"])

    # frustratingly simple
    # print "Running augment"
    # augment = domain_adaptation.run_frustratingly_simple(model, Xt, yt, lt, Xs, ys, ls, grid)
    # augment_disc = domain_adaptation.run_frustratingly_simple(model, Xt_disc, yt_disc, lt_disc, Xs_disc, ys_disc, ls_disc, grid)
    # save_csv("output/tmp_csvs/augment.csv",[augment_disc, augment], ["augment_with_disc", "augment_without_disc"], ticks=grid)
    # plot_results([augment_disc, augment], ["augment_with_disc", "augment_without_disc"], with_err=True, ticks=grid)
    

    # Coral
    # print "Running CORAL"
    
    # domain_adaptation.runPCA(Xt,Xs)

    # coral_prec = domain_adaptation.run_coral(model, Xt, yt, lt, Xs, ys, ls, grid, precision_score)
    # coral_rec = domain_adaptation.run_coral(model, Xt, yt, lt, Xs, ys, ls, grid, recall_score)
    # coral_f1 = domain_adaptation.run_coral(model, Xt, yt, lt, Xs, ys, ls, grid, f1_score)
    # coral = domain_adaptation.run_coral(model, Xt, yt, lt, Xs, ys, ls, grid, f1_score)
    # coral_disc = domain_adaptation.run_coral(model, Xt_disc, yt_disc, lt_disc, Xs_disc, ys_disc, ls_disc, grid, f1_score)
    # plot_results([coral], ["coral_without_disc"], with_err=True, ticks=grid)
    # plot_results([coral_disc, coral], ["coral_with_disc", "coral_without_disc"], with_err=True, ticks=grid)
    # save_csv("output/tmp_csvs/coral_normalized.csv",[coral_disc, coral], ["coral_with_disc", "coral_without_disc"])



    # plot_results([coral_prec, coral_rec, coral_f1, ], ["CORAL_precison","CORAL_recall","CORAL_f1",], with_err=True)
    # # plot_bar([bl_1, bl_2, bl_3, bl_4, frus, coral], ["bl_1","bl_2","bl_3","bl_4", "frus", "coral"])
    # plot_results([bl_1, bl_2, bl_3, bl_4, frus, coral], ["bl_1","bl_2","bl_3","bl_4", "frus", "coral"], with_err=False)
    # # domain_adaptation.get_top_correlated_features_frust(Xt, yt, Xs, ys)


# ======================================================================================================
if __name__ == '__main__':
    # models = {}
    # for file in os.listdir("../docs/output/halves/csvs/"):
    #     if file.endswith("baseline_LogisticRegression.csv"):
    #          + path = "../docs/output/halves/csvs/"file
    #         division = file.split("_")[0]
    #         model = file.split("_")[1].split(".csv")[0]
    #         mean, std = get_max_std(path)
    #         print "%s, %s --> %.3f, %.3f, " % (model,division,mean,std)

    # print_stats("baseline_LogisticRegression")
    # print_stats("halves_LogisticRegression")
    # print_stats("strips_LogisticRegression")
    # print_stats("quadrants_LogisticRegression")
    # run_experiment(LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT))


    run_domain_adaptation()
    # majority_class_with = pd.read_csv("output/tmp_csvs/majority_class.csv")
    # majority_class_without = pd.read_csv("output/tmp_csvs/majority_class.csv")
    # majority_class_with['Discourse Features'] = "Including Discourse Features"
    # majority_class_without['Discourse Features'] = "Excluding Discourse Features"

    # target_only    = pd.read_csv("output/tmp_csvs/target_only.csv")
    # source_only    = pd.read_csv("output/tmp_csvs/source_only.csv")
    # relabeled      = pd.read_csv("output/tmp_csvs/relabeled.csv")
    # augment        = pd.read_csv("output/tmp_csvs/augment_hand_modified.csv")
    # coral          = pd.read_csv("output/tmp_csvs/coral.csv")

    # dfs = pd.concat([majority_class_with,majority_class_without, target_only, source_only, relabeled, augment, coral])
    # dfs.ix[dfs.model.str.contains('_with_disc'), 'Discourse Features'] = "Including Discourse Features"
    # dfs.ix[dfs.model.str.contains('_without_disc'), 'Discourse Features'] = "Excluding Discourse Features"
    # dfs['model'] = dfs['model'].map(lambda x: x.replace('_with_disc','').replace('_without_disc',''))
    
    # dfs.replace("bl_0", "majority_class", inplace=True)

    # # for m in dfs.model.unique():  
    # #     v1 = dfs[(dfs.model == m) & (dfs['Discourse Features'] == "Excluding Discourse Features")].fold_vals
    # #     c1 = st.t.interval(0.90, len(v1)-1, loc=np.mean(v1), scale=st.sem(v1))
    # #     v2 = dfs[(dfs.model == m) & (dfs['Discourse Features'] == "Including Discourse Features")].fold_vals
    # #     c2 = st.t.interval(0.90, len(v2)-1, loc=np.mean(v2), scale=st.sem(v2))
    # #     print "%s w/o discourse: %.7f 90  CI [%.7f, %.7f ]" % (m, np.mean(v1), c1[0], c1[1])
    # #     print "%s w discourse: %.7f 90  CI [%.7f, %.7f ]" % (m, np.mean(v2), c2[0], c2[1])


    # # sns.set_style("whitegrid")
    # sns.set(font_scale=1.8)
    # sns.plt.ylim(0, 1)
    # ax = sns.barplot(x="model", y="fold_vals", hue='Discourse Features', data=dfs, ci=90, errwidth=1.25,  capsize=.2)
    # ax.set(xlabel='Model', ylabel='F-Measure')
    # ax.figure.tight_layout()
    # sns.set_style("whitegrid")
    # sns.plt.show()

    # # ax.figure.savefig("output/plots/winter_2017/model_comparison.png")
    
    
    # dfs = dfs[['best_k', 'model', 'Discourse Features']].drop_duplicates()
    # dfs = dfs[dfs.model != "majority_class"]

    # ncols_disc = 353
    # ncols_nodisc = 313
    # dfs['feature_ratio'] = 0
    # dfs.ix[dfs['Discourse Features'] == 'Excluding Discourse Features', "feature_ratio"] = dfs.ix[dfs['Discourse Features'] == 'Excluding Discourse Features', "best_k"] / ncols_nodisc
    # dfs.ix[dfs['Discourse Features'] == 'Including Discourse Features', "feature_ratio"] = dfs.ix[dfs['Discourse Features'] == 'Including Discourse Features', "best_k"] / ncols_disc
    # dfs.ix[dfs['model'] == 'augment', "feature_ratio"] = dfs.ix[dfs['model'] == 'augment', "feature_ratio"]/3

    # ax = sns.barplot(x="model", y="feature_ratio", hue='Discourse Features', data=dfs)
    # ax.set(xlabel='Model', ylabel='% of features selected ')
    # ax.figure.tight_layout()
    # ax.figure.savefig("output/plots/winter_2017/feature_count.png")
    # sns.plt.show()
    # 