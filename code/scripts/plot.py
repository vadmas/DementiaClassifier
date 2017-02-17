import pandas as pd
import seaborn as sns

# --------MySql---------
import db
cnx = db.get_connection()
# ----------------------

# Table names
MAJORITY_CLASS         = 'cv_majority_class'
AUGMENT_WITH_DISC      = 'cv_augment_with_disc_from_100_to_900_by_1'
AUGMENT_WITHOUT_DISC   = 'cv_augment_without_disc_from_100_to_900_by_1'
CORAL_WITH_DISC        = 'cv_coral_with_disc_from_1_to_350_by_1'
CORAL_WITHOUT_DISC     = 'cv_coral_without_disc_from_1_to_350_by_1'
RELABELED_WITH_DISC    = 'cv_relabeled_with_disc_from_1_to_350_by_1'
RELABELED_WITHOUT_DISC = 'cv_relabeled_without_disc_from_1_to_350_by_1'
SOURCE_ONLY_WITH_DISC       = 'cv_source_only_with_disc_from_1_to_350_by_1'
SOURCE_ONLY_WITHOUT_DISC    = 'cv_source_only_without_disc_from_1_to_350_by_1'
TARGET_ONLY_WITH_DISC       = 'cv_target_only_with_disc_from_1_to_350_by_1'
TARGET_ONLY_WITHOUT_DISC    = 'cv_target_only_without_disc_from_1_to_350_by_1'


def print_stats(table_name):
    table = pd.read_sql_table(table_name, cnx, index_col='index')
    max_k = table.mean().argmax()
    print '%s:\n %f +/- %f, max_k: %s' % (table_name, table[max_k].mean(), table[max_k].std(), max_k)


def get_max_fold(table_name):
    table = pd.read_sql_table(table_name, cnx, index_col='index')
    max_k = table.mean().argmax()
    df = table[max_k].to_frame()
    df.columns = ['folds']
    return df


def barplot():
    data = {
        'Majority_Class': [MAJORITY_CLASS, MAJORITY_CLASS],
        'Augment': [AUGMENT_WITH_DISC, AUGMENT_WITHOUT_DISC],
        'Coral': [CORAL_WITH_DISC, CORAL_WITHOUT_DISC],
        'Relabeled': [RELABELED_WITH_DISC, RELABELED_WITHOUT_DISC],
        'Source_Only': [SOURCE_ONLY_WITH_DISC, SOURCE_ONLY_WITHOUT_DISC],
        'Target_Only': [TARGET_ONLY_WITH_DISC, TARGET_ONLY_WITHOUT_DISC],
    }

    dfs = []
    models = ['Majority_Class', 'Target_Only', 'Source_Only', 'Relabeled', 'Augment', 'Coral', ]
    for model in models:
        withdisc    = get_max_fold(data[model][0])
        withoutdisc = get_max_fold(data[model][1])
        withoutdisc['Model'] = model
        withdisc['Model']    = model
        withoutdisc['Discourse Features'] = 'Excluding Discourse Features'
        withdisc['Discourse Features']    = 'Including Discourse Features'
        dfs.append(withoutdisc)
        dfs.append(withdisc)

    dfs = pd.concat(dfs)

    sns.set_style('whitegrid')
    sns.set(font_scale=1.8)
    sns.plt.ylim(0, 1)
    ax = sns.barplot(x='Model', y='folds', hue='Discourse Features', data=dfs, ci=90, errwidth=1.25, capsize=.2)
    ax.set(xlabel='Model', ylabel='F-Measure')
    ax.figure.tight_layout()
    sns.set_style('whitegrid')
    sns.plt.show()


if __name__ == '__main__':
    barplot()
