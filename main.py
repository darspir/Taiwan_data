# bibliographies ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import own_process_data as opd
from src import own_classification as oc
from math import exp, log


# main -------------------------------------------------------------------------
if __name__ == "__main__":

    # read the data ------------------------------------------------------------
    regressor_ind = range(24)
    explained_var_ind = [24]
    splitpercentage = 0.9

    XX, Y, XX_test, Y_test = opd.read_and_split("default of credit card clients.xls", regressor_ind, explained_var_ind, splitpercentage)

    # summary statistics
    pd.DataFrame(pd.read_excel("default of credit card clients.xls", skiprows = 1, nrows = 30000, usecols = range(24))).describe().to_latex(r"summary_statistics.tex", float_format="%.2f", longtable = True)

    # build matrix where CLASSES[0] is all different labels (binary = [0, 1])
    # CLASSES[1][j] contains all rownumbers (observation) of label j
    print("\ntraining set:")
    CLASSES, descrX = opd.describe_data(Y)
    print("\ntest set:")
    CLASSES_test, descrX = opd.describe_data(Y_test)

    # choose important features via heatmap (correlation matrix)
    # Compute correlation matrix
    M = np.append(XX, Y, axis = 1)
    M_dataframe = pd.DataFrame(M)

    correlation_matrix =  np.array(M_dataframe.corr(method = 'pearson'))
    regressor_labels = descrX + ["Y"]

    # Plot heatmap of correlation matrix
    opd.plot_pearson(correlation_matrix, regressor_labels)

    # heatmap can be seen in folder
    # result: one could choose here the entries with the biggest absolute values for correlation between regressor and explained variable and small absolute values of correlation between regressors

    classes = [0, 1]

    Classes = opd.select_classes(XX, CLASSES, classes)
    Classes_test = opd.select_classes(XX_test, CLASSES_test, classes)
    if len(Classes[0]) != len(Classes_test[0]):
        exit("Training and testing set unequally splitted. Choose another seed and/or percentage.")

    # use the following features in classification task
    feature_ind = range(24)

    # normalize data (see test set with parameters of training)
    X, mu, sigma = opd.normalize(XX[:, feature_ind])
    X_test = (XX_test[:, feature_ind] - mu)/sigma

    # X matrix for classification (including column of 1's for the intercept)
    X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis = 1)


################################################################################


    # use logistic classification and train model
    print("\nLOGISTIC CLASSIFICATION")
    # set up dictionary to train model
    # k specifies if training on whole data set (k=1) | k-fold (k = k) |
    # or LOOCV (k = X.shape[0])
    # we specify the method (though we only implemented one) for future
    # applications, where one might use another method (CV and minimization etc.
    # would still work)

    # !!!! BE AWARE: the cross-validation takes ages to compute (epsecially
    # choosing a big k)... to verify that code works you may wanna try it with
    # k = 2
    dict = {
        "X": X,
        "Y": Y,
        "k_0": 1,
        "stoch": False,
        "alpha_grid": [i * 5e-3 for i in range(3,4)],
        "C_grid": np.arange(0.1,0.6,0.01),
        "method": "log_cla"
    }


    dict["beta"], dict["hyper_para"] = oc.get_parameters(**dict)
    dict["C"] = dict["hyper_para"][2]

    print("\nRESULT:")

    print("\nhyper_para:")
    print(dict["hyper_para"])

    print("\nbeta:")
    print(dict["beta"])

    print("\nFNR training:")
    print(oc.loss(**dict).FNR(**dict))

    print("\nConfusion Matrix for training:\n")
    oc.loss(**dict).print_confusion_matrix(**dict)

    print("\ntraining loss: \
            {}".format(oc.loss(**dict).los.calculate_loss(**dict)))

    # Verify predictive power of classifier on the test set
    dict["X"], dict["Y"] = X_test, Y_test

    print("\nFNR test:")
    print(oc.loss(**dict).FNR(**dict))

    print("\nConfusion Matrix for test:\n")
    oc.loss(**dict).print_confusion_matrix(**dict)

    print("\ntest loss: {}".format(oc.loss(**dict).los.calculate_loss(**dict)))

    print("\nACC for test: {}".format(oc.loss(**dict).ACC(**dict)))

    dict["p_pred"] = oc.predict(**dict)[1]
    oc.SSM(**dict).plot_prob_vs_prob()

################################################################################
