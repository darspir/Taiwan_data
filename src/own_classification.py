import numpy as np
from math import exp, log


def predict(X, beta, hyper_para = [1e-4, 1e-1, 0.5], **kwargs):
    kwargs["X"] = X
    kwargs["beta"] = beta
    if "C" in kwargs:
        C = kwargs["C"]
    else:
        C = hyper_para[2]

    eta = np.zeros((X.shape[0], 1))
    f = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        eta[i] = 1 / (1 + exp(-X[i,:] @ beta))
        if eta[i] - C > 0:
            f[i] = 1
    return f


class loss:
    "lossfunctions"
    def __init__(self, los = "log", **kwargs):
        if los == "log":
            self.los = self.log
        else:
            self.los = self.log

    class log:
        "loss and gradient for logistic classification"

        def calculate_loss(X, Y, beta, **kwargs):
            kwargs["X"] = X
            kwargs["Y"] = Y
            kwargs["beta"] = beta
            n = X.shape[0]
            loss = 0
            for i in range(n):
                loss += - 1/n * ( Y[i] * (X[i,:] @ beta) - log(1 + exp(X[i,:] @ beta), 2) )
            return loss

        def beta_grad(X, Y, beta, C, Lambda = 10**(-4), **kwargs):
            kwargs["X"] = X
            kwargs["Y"] = Y
            kwargs["beta"] = beta
            kwargs["C"] = C
            kwargs["Lambda"] = Lambda

            grad = np.zeros((X.shape[1], 1))

            for i in range(X.shape[0]):
                grad += 1/X.shape[0] * X[i,:].reshape((X.shape[1], 1)) * (1/ (1 + exp(-X[i,:] @ beta)) - Y[i])


            return grad


class gd(loss):
    "NADAM"
    def __init__(self, X, Y,  hyper_para = [1e-4, 1.5*1e-2, 0.5], batch_number = 10,
                    los = "l2", max_iter = 1e4, eps = 1e-6, memory_size = 50,
                    eta1 = 0.9, eta2 = 0.999, e = 1e-8, **kwargs):
        loss.__init__(self, los, **kwargs)
        self.X = X
        self.Y = Y
        if "Lambda" in kwargs:
            self.Lambda = kwargs["Lambda"]
        else:
            self.Lambda = hyper_para[0]
        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]
        else:
            self.alpha = hyper_para[1]
        if "C" in kwargs:
            self.C = kwargs["C"]
        else:
            self.C = hyper_para[2]
        self.batch_number = batch_number
        self.max_iter = int(max_iter)
        self.eps = eps
        self.memory_size = memory_size
        if "beta_init" not in kwargs:
            self.beta = np.zeros((self.X.shape[1], 1))
        else:
            self.beta = kwargs["beta_init"]
        self.m = np.zeros((self.X.shape[1], 1))
        self.v = np.zeros((self.X.shape[1], 1))
        self.eta1 = eta1
        self.eta2 = eta2
        self.e = e



    def grad_descent(self):
        memory = np.zeros(self.memory_size)
        for i in range(self.max_iter):
            x = self.beta
            self.m = self.eta1 * self.m + (1 - self.eta1)\
                    * self.los.beta_grad(self.X, self.Y, self.beta, self.C,
                                                            self.Lambda)
            mhat = self.m / (1-self.eta1**(i+1))
            self.v = self.eta2 * self.v + (1-self.eta2)\
                    * (self.los.beta_grad(self.X, self.Y, self.beta, self.C,
                                                            self.Lambda))**2
            vhat = self.v / (1-self.eta2**(i+1))
            self.beta = self.beta - self.alpha / (np.sqrt(vhat) + self.e)\
                    * (self.eta1 * mhat + (1-self.eta1)\
                    * self.los.beta_grad(self.X, self.Y, self.beta, self.C,
                                                            self.Lambda)\
                    / (1-self.eta1**(i+1)))
            memory[i % self.memory_size] = np.sum(np.abs(self.beta - x))
            if i > self.memory_size and np.sum(memory) < self.eps:
                break

#ordinary gradient descent
#        for i in range(self.max_iter):
#            self.beta = self.beta - self.alpha * self.los.beta_grad(self.X, self.Y, self.beta, self.C, self.Lambda).reshape((self.X.shape[1], 1))

        return self.beta


    def stoch_grad_descent(self):

        n,m = self.X.shape

        self.batch_number = min(abs(self.batch_number), n)
        self.batch_number = max(self.batch_number, 1)

        random_index = np.random.choice(self.batch_number, self.batch_number,
                                            replace = False)

        memory = np.zeros(self.memory_size)
        for i in range(self.max_iter):

            first_cut = random_index[i%self.batch_number]\
                        * int(n / self.batch_number)
            second_cut = random_index[i%self.batch_number]\
                        * int(n / self.batch_number)\
                        + int(n / self.batch_number)

            X_batch = self.X[first_cut:second_cut, :]
            Y_batch = self.Y[first_cut:second_cut,:]
            x = self.beta
            self.m = self.eta1 * self.m + (1 - self.eta1)\
                        * self.los.beta_grad(X_batch, Y_batch, self.beta,
                                                            self.Lambda)
            mhat = self.m / (1-self.eta1**(i+1))
            self.v = self.eta2 * self.v + (1-self.eta2)\
                        * (self.los.beta_grad(X_batch, Y_batch, self.beta,
                                                            self.Lambda))**2
            vhat = self.v / (1-self.eta2**(i+1))
            self.beta = self.beta - self.alpha / (np.sqrt(vhat) + self.e)\
                        * (self.eta1 * mhat + (1-self.eta1)\
                        * self.los.beta_grad(X_batch, Y_batch, self.beta,
                                                            self.Lambda)\
                        / (1-self.eta1**(i+1)))
            memory[i % self.memory_size] = np.sum(np.abs(self.beta - x))
            if i > self.memory_size and np.sum(memory) < self.eps:
                break

            if (i+1) % (self.batch_number) == 0:
                random_index = np.random.choice(self.batch_number, self.batch_number, replace = False)


        return self.beta


def general_get_beta(method = "log", stoch = False, **kwargs):

    GD = gd(**kwargs)
    if stoch == True:
        return GD.stoch_grad_descent()
    else:
        return GD.grad_descent()



def get_parameters(X, Y, k_0 = 1, method = "log", **kwargs):
    """
    Parameters
    ----------
    X : array
        (n,k) array of explanatory variables including a column of ones for the
        intercept
    Y : array
        (n,1) vector of dependent variables
    k_0 = int > 0 (will be automatic corrected if false)
        number of folds when searching the best hyperparameters
        has to be smaller than n + 1 where n is the number of lines of X (this
        will automatically be corrected by min(k_0, n))
    lossfunction_key = "lossfunction":
            lossfunction in {log}

    Returns
    -------
    get_beta_lossfunction()
    hyper_para: array
        best hyperparameters for get_beta_lossfunction()
    """

    kwargs["method"] = method


    n, m = X.shape

    k_0 = abs(k_0)
    k_0 = max(k_0, 1)
    k_0 = min(k_0, n)

    # add degree_grid
    #default hyperparameters
    if "Lambda_grid" not in kwargs:
        kwargs["Lambda_grid"] = [10**(-4)]
    hyperparameters = [kwargs["Lambda_grid"]]


    if "alpha_grid" not in kwargs:
        kwargs["alpha_grid"] = [10**(-2)]
    hyperparameters.append(kwargs["alpha_grid"])

    if "C_grid" not in kwargs:
        kwargs["C_grid"] = [0.5]
    hyperparameters.append(kwargs["C_grid"])

    hyper_number = len(hyperparameters)

    #create grid of the hyperparameters
    hyper_grid = np.array(np.meshgrid(*hyperparameters)).T.reshape(-1, hyper_number)


    # find the best hyperparameters

    if k_0 > 1:   # do CV over k_0 folds

        cv_error = np.zeros((k_0,1))
        CV_error = 0
        hyper_para = []

        # calculate possible results with the points in the grid
        for i in range(0, len(hyper_grid)):
            kwargs["hyper_para"] = hyper_grid[i][:]

            # do the CV
            for j in range(k_0):

                first_cut = j * int(n / k_0)
                second_cut = j * int(n / k_0) + int(n / k_0)

                kwargs["X"] = np.delete(X, slice(first_cut, second_cut),axis=0)
                kwargs["Y"] = np.delete(Y, slice(first_cut, second_cut),axis=0)

                kwargs["beta"] = general_get_beta(**kwargs)

                kwargs["X"] = X[first_cut:second_cut,:]
                kwargs["Y"] = Y[first_cut:second_cut]

                cv_error[j] = loss(**kwargs).los.calculate_loss(**kwargs)


            new_CV_error = np.mean(cv_error)

            # check whether new grid-point is better, and replace the results
            # if it is true
            if new_CV_error < CV_error or i == 0:
                CV_error = new_CV_error
                hyper_para = hyper_grid[i][:]

        kwargs["hyper_para"] = hyper_para

        kwargs["X"] = X
        kwargs["Y"] = Y
        beta = general_get_beta(**kwargs)

    else:   #no CV

        error = 0
        hyper_para = []

        kwargs["X"] = X
        kwargs["Y"] = Y

        # calculate possible results with the points in the grid
        for i in range(0, len(hyper_grid)):

            kwargs["hyper_para"] = hyper_grid[i][:]

            kwargs["beta"] = general_get_beta(**kwargs)

            new_error = loss(**kwargs).los.calculate_loss(**kwargs)

            # check whether new grid-point is better, and replace the results
            # if it is true
            if new_error < error or i == 0:
                error = new_error
                hyper_para = hyper_grid[i][:]


        kwargs["hyper_para"] = hyper_para


        beta = general_get_beta(**kwargs)


    return beta, hyper_para
