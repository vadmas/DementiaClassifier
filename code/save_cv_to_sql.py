import pandas as pd
from scripts.domain_adaptation import DomainAdapter
from sklearn.metrics import f1_score

# models
from sklearn.linear_model import LogisticRegression

# --------MySql---------
import db
cnx = db.get_connection()
# ----------------------

REGULARIZATION_CONSTANT = 1


def save_domain_adaptation_to_sql_helper(da, method, name, kmin, kmax, interval):
    k_range = range(kmin, kmax, interval)
    da.train_model(method, k_range)
    df = pd.DataFrame(da.results[method], columns=k_range)
    name = "cv_%s_%s_from_%i_to_%i_by_%i" % (method, name, kmin, kmax, interval)
    df.to_sql(name, cnx, if_exists='replace')


def save_domain_adaptation_to_sql():
    model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)
    da_with_disc = DomainAdapter(model, f1_score, with_disc=True)
    da_without_disc = DomainAdapter(model, f1_score, with_disc=False)

    da_with_disc.train_majority_class()
    import pdb
    pdb.set_trace()
    save_domain_adaptation_to_sql_helper(da_with_disc, 'majority_class', '', 1, 1, 1)
    import pdb
    pdb.set_trace()

    # # target_only
    # print "Running target_only..."
    # save_domain_adaptation_to_sql_helper(da_with_disc,'target_only','with_disc',1,350,1)
    # save_domain_adaptation_to_sql_helper(da_without_disc,'target_only','without_disc',1,350,1)

    # # source_only
    # print "Running source_only..."
    # save_domain_adaptation_to_sql_helper(da_with_disc,'source_only','with_disc',1,350,1)
    # save_domain_adaptation_to_sql_helper(da_without_disc,'source_only','without_disc',1,350,1)

    # # relabeled
    # print "Running relabeled..."
    # save_domain_adaptation_to_sql_helper(da_with_disc,'relabeled','with_disc',1,350,1)
    # save_domain_adaptation_to_sql_helper(da_without_disc,'relabeled','without_disc',1,350,1)

    # # augment
    print "Running augment..."
    save_domain_adaptation_to_sql_helper(da_with_disc, 'augment', 'with_disc', 100, 900, 1)
    save_domain_adaptation_to_sql_helper(da_without_disc, 'augment', 'without_disc', 100, 900, 1)

    # # coral
    # print "Running coral..."
    # save_domain_adaptation_to_sql_helper(da_with_disc,'coral','with_disc',1,350,1)
    # save_domain_adaptation_to_sql_helper(da_without_disc,'coral','without_disc',1,350,1)


if __name__ == '__main__':
    save_domain_adaptation_to_sql()
