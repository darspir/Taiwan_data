# bibliographies ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import own_process_data as opd
from src import own_classification as oc
from math import exp, log

#-------------------------------------------------------------------------------

# plotting function ------------------------------------------------------------

def plot_surface(x, y, z, azim=-60, elev=40, dist=10, cmap="RdYlBu_r"):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_args = {'rstride': 1, 'cstride': 1, 'cmap':cmap,
             'linewidth': 20, 'antialiased': True,
             'vmin': -2, 'vmax': 2}
    ax.plot_surface(x, y, z, **plot_args)
    ax.view_init(azim=azim, elev=elev)
    ax.dist=dist
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    plt.xticks([-1, -0.5, 0, 0.5, 1], ["-1", "-1/2", "0", "1/2", "1"])
    plt.yticks([-1, -0.5, 0, 0.5, 1], ["-1", "-1/2", "0", "1/2", "1"])
    ax.set_zticks([-2, -1, 0, 1, 2])
    ax.set_zticklabels(["-2", "-1", "0", "1", "2"])

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_zlabel("z", fontsize=18)
    return fig, ax;



# main -------------------------------------------------------------------------
if __name__ == "__main__":

    # read the data ------------------------------------------------------------
    regressor_ind = range(20)
    explained_var_ind = [20]
    splitpercentage = 0.9

    X, Y, X_test, Y_test = opd.read_and_split("SouthGermanCredit.asc", regressor_ind, explained_var_ind, splitpercentage)

    # build matrix where CLASSES[0] is all different labels (binary = [0, 1])
    # CLASSES[1][j] contains all rownumbers (observation) of label j
    print("\ntraining set:")
    CLASSES, descrX = opd.describe_data(Y)
    print("\ntest set:")
    CLASSES_test, descrX = opd.describe_data(Y_test)


    # choose important features via heatmap (correlation matrix)
    # Compute correlation matrix
    M = np.append(X, Y, axis = 1)
    M_dataframe = pd.DataFrame(M)

    correlation_matrix =  np.array(M_dataframe.corr(method = 'pearson'))
    regressor_labels = descrX + ["kredit"]

    # Plot heatmap of correlation matrix
    opd.plot_pearson(correlation_matrix, regressor_labels)
    # heatmap can be seen in folder
    # result: we see that laufkont, laufzeit, weitkred have big
    # absolute correlation to the explained variable (kredit), but no big
    # correlation between themselves
    # => we choose this 3 regressors for the following classification tasks
    classes = [0, 1]


    Classes = opd.select_classes(X, CLASSES, classes)
    Classes_test = opd.select_classes(X_test, CLASSES_test, classes)
    if Classes[0] != Classes_test[0]:
        exit("Training and testing set unequally splitted. Choose another seed and/or percentage.")

    feature_ind = [i for i in range(len(descrX)) if ('laufkont' == descrX[i] or\
    'laufzeit' == descrX[i] or 'weitkred' == descrX[i] or 'moral' == descrX[i] or 'hoehe' == descrX[i] or 'sparkont' == descrX[i])]

    #X, mu, sigma = opd.normalize(X[:, feature_ind])
    #X_test = (X_test[:, feature_ind] - mu)/sigma

    X = X[:, feature_ind]
    X_test = X_test[:, feature_ind]

#    # visualize classes in dependence of one features
#    fig = plt.figure(figsize = (10,5))
#    plt.rc('font', size=14)
#    plt.rc('xtick', labelsize=14)

#    ax = fig.add_subplot()

#    binwidth = 0.1


#    for i in range(len(classes)):
#        x = X[Classes[1][i]]
#        ax.hist(x, bins = np.arange(min(x), max(x)+binwidth, binwidth), label = "class %d"%i, density = True)


#    ax.grid()

#    ax.set_title("Histogram of the feature")

#    handles, labels = ax.get_legend_handles_labels()
#    legcols = 1
#    ax.legend(handles, labels,loc = 'best', markerscale = 3, ncol = legcols)

#    fig.savefig("Histogram.png")


    # use logistic classification (hyperparameter via CV) ----------------------
    # X matrix for classification (including column of 1's for the intercept)
    X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis = 1)

    dict = {
        "X": X,
        "Y": Y,
        "k_0": 1,
        "stoch": False,
        "C_grid": [0.05*i for i in range(10, 11)],
        "alpha": 0.1,
        "beta_init": np.array([2,4,-3,1,0,10,-6]).reshape((7,1))
    }

#    def cost(b0, b1):
#        n = X.shape[0]
#        loss = 0
#        for i in range(n):
#            loss += - 1/n * ( Y[i] * (X[i,0] *b0 + X[i,1] *b1) - log(1 + exp(X[i,0] *b0 + X[i,1] *b1), 2) )
#        return loss
#    Cost = np.vectorize(cost)

#    plt.close()
#    x, y = np.mgrid[-2:2:40j, -2:2:40j]
#    plt.plot(-0.72, 0.64 , Cost(-0.72, 0.64), marker = "o", markersize = 20)
#    fig0,ax0=plot_surface(x, y, Cost(x, y))
#    ax0.set_xlabel("$\omega_1$")
#    ax0.set_ylabel("$\omega_2$")
#    ax0.set_zlabel("$C_1(\omega)$")
#    ax0.tick_params(axis='both', labelsize=10)
#    plt.tight_layout()
#    plt.show()
#    plt.close()



    dict["beta"], dict["hyper_para"] = oc.get_parameters(**dict)

    print("\nbeta:")
    print(dict["beta"])

    print("\nhyper_para:")
    print(dict["hyper_para"])

    print("\nprediciton - label")
    print(np.sum(abs(oc.predict(**dict).transpose() - Y.transpose())))

    print("\ntraining loss")
    print(oc.loss(**dict).los.calculate_loss(**dict))



    # classification (lets see how the features are distributed, if approx gaussian -> LDA/QDA)