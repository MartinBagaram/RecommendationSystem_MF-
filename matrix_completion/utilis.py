import numpy as np

def test_train_split(train_df, seed=1, min_sel=1):
    """

    """
    np.random.seed(seed)
#     picked = set()
    dict_ = {k: (~train_df[k].isnull()).sum() for k in train_df.columns}
    sorted_d = sorted(dict_.items(), key=lambda x: x[1])
    train = train_df.copy()
    test = train_df.copy()
    test = test * 0
    for col , num in sorted_d:
        if num > min_sel:
            full_col = train[col]
            indices = train[~full_col.isnull()].index
            # only rows for that column that were not already selected before
            # only keep rows that have more than one observation
            indices = [ind for ind in indices if np.count_nonzero(list(train_df.loc[ind, :])) > 1 ]
            if len(indices) > 0:
                pick = np.random.choice(indices, size = max(1, len(indices)//10))
                for p in pick:
                    test.loc[p, col] = train.loc[p, col]
                    train.loc[p, col] = 0
                
    train.fillna(0.0, inplace=True)
    test.fillna(0.0, inplace=True)
    return test, train

def is_split_good(train):
    """
    """
    for i in range(np.shape(train)[0]):
        if np.count_nonzero(train[i,:]) == 0:
            return False
    for j in range(np.shape(train)[1]):
        if np.count_nonzero(train[:,j]) == 0:
            return False
    return True