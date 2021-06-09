import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def split_sample(sample, train_size, random = False):

    full_sample = np.array(sample)
    n, k = full_sample.shape
    split_index = int(round(train_size * n, 0))

    if random == False:
        #split the sample
        train_sample = full_sample[0:split_index,:]
        test_sample = full_sample[split_index:,:]

    else:
        #permute the full sample
        full_sample_permuted = np.zeros((n, k))
        np.random.seed(1)
        random_index = np.random.choice(n,n, replace = False)
        for i in range(0,n):
            full_sample_permuted[i, :] = full_sample[random_index[i],:]

        #split the sample
        train_sample = full_sample_permuted[0:split_index,:]
        test_sample = full_sample_permuted[split_index:,:]

    return train_sample, test_sample


def read_and_split(dataset, regressor_ind, explained_var_ind,
                splitpercentage = 0.9, random = True):

    Data = np.array(pd.read_excel(dataset, skiprows = 1, nrows = 30000))

    train_sample, test_sample = split_sample(Data, splitpercentage, random = random)


    n, k = train_sample.shape
    m, q = test_sample.shape

    X_train = train_sample[:, regressor_ind].reshape((n,len(regressor_ind)))
    Y_train = train_sample[:, explained_var_ind].reshape((n,1))

    X_test = test_sample[:, regressor_ind]
    Y_test = test_sample[:, explained_var_ind]

    return  X_train,  Y_train,  X_test, Y_test


def describe_data(Y):

    n = np.array(Y).shape[0]

    # create Classes[0] = name of all classes, Classes[1][j] = number (row) of observation in class j
    Classes = [[], []]
    for i in range(0, n):
        if Y[i] not in Classes[0]:
            Classes[0].append(Y[i])
            Classes[1].append([])
            Classes[1][len(Classes[0])-1].append(i)

        else:
            for j in range(len(Classes[0])):
                if Y[i] == Classes[0][j]:
                    (Classes[1][j]).append(i)
                    break

    descrX = ['X1', 'X2', 'X3', 'X4',
            'X5', 'X6', 'X7', 'X8',
            'X9', 'X10', 'X11', 'X12', 'X13',
            'X14', 'X15', 'X16', 'X17', 'X18',
            'X19', 'X20', 'X21', 'X22', 'X23', 'X24']

    for i in range(len(Classes[0])):
        print("Observations in class %d: %d"%(i, len(Classes[1][i])))

    return Classes, descrX


def plot_pearson(correlation_matrix, regressor_labels):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(abs(correlation_matrix), cmap="YlGnBu")


    ax.set_xticks(np.arange(len(regressor_labels)))
    ax.set_yticks(np.arange(len(regressor_labels)))
    ax.set_xticklabels(regressor_labels)
    ax.set_yticklabels(regressor_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(regressor_labels)):
        for j in range(len(regressor_labels)):
            text = ax.text(j, i, round(correlation_matrix[i, j],2),
                           ha="center", va="center", color="k")

    ax.set_title("Heatmap of Correlation Matrix")

    fig.tight_layout(pad=1)

    plt.savefig("heatmap.png")


def select_classes(X, CLASSES, classes):
    return [[CLASSES[0][e] for e in classes], [CLASSES[1][e] for e in classes]]


def normalize(X):

    n, k = X.shape
    mu = np.zeros((1, k))
    sigma = np.zeros((1, k))

    for i in range (0,k):
        mu[0,i] = np.mean(X[:,i].reshape((n,1)))
        sigma[0,i] = np.sqrt(np.var(X[:,i].reshape((n,1))))

    return (X - mu)/sigma, mu, sigma


def subtract_mean(Y):

    n, k = Y.shape
    mu = np.zeros((1, k))

    for i in range (0,k):
        mu[0,i] = np.mean(X[:,i].reshape((n,1)))

    return Y - mu, mu
