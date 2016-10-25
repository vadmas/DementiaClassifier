import numpy as np
import pandas as pd
from sklearn.utils import shuffle

ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ['Control']
NON_ALZHEIMERS = ["MCI", "Memory", "Other", "Vascular"]


# Helper function to map row to new feature space
# in accordance with 'frustratingly simple' paper
def triplicate_row(row,target=True):
    vals = row.values.tolist()
    zeros = np.zeros_like(row).tolist()
    if target:
        return vals + zeros + vals
    else:
        return vals + vals + zeros


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
    import ipdb; ipdb.set_trace()
    print "test"
    print "test"
