import numpy as np
def make_z_score_data(train,test):
    test = test.values
    for m in range(train.shape[1]):
        Mean = np.mean(train[:,m])
        STD = np.std(train[:,m])

        for n in range(test.shape[0]):
            test[n, m] = (test[n, m] - Mean) / STD

    return test


