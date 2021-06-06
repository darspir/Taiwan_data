import numpy as np
from math import exp, log


def predict(X, beta, method = "log_cla", hyper_para = [1e-4, 1e-2, 0.5], **kwargs):
    kwargs["X"] = X
    kwargs["beta"] = beta
    if "C" in kwargs:
        C = kwargs["C"]
    else:
        C = hyper_para[2]
    kwargs["method"] = method

    if method == "log_reg":
        eta = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            eta[i] = 1 / (1 + exp(-X[i,:] @ beta))
        return eta

    else:
        eta = np.zeros((X.shape[0], 1))
        f = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            eta[i] = 1 / (1 + exp(-X[i,:] @ beta))
            if eta[i] - C > 0:
                f[i] = 1
        return f

def dec_boundary(Y, C):
    f = np.zeros((Y.shape[0], Y.shape[1]))
    for i in range(Y.shape[0]):
        if Y[i] > C:
            f[i,:] = 1
    return f

class loss:
    "lossfunctions"
    def __init__(self, method = "log_cla", **kwargs):
        kwargs["method"] = method
        if method == "log_reg":
            self.los = self.log_reg
        else:
            self.los = self.log_cla


    def TP(self, X, Y, hyper_para = [1e-4, 1e-2, 0.5], method = "log_cla", **kwargs):
        kwargs["X"] = X
        kwargs["Y"] = Y
        if "C" in kwargs:
            C = kwargs["C"]
        else:
            C = hyper_para[2]
        if method == "log_cla":
            Prediction = predict(**kwargs)
        else:
            Prediction = dec_boundary(predict(**kwargs), C)
        n = 0
        tp = 0
        for i in [k for k in range(X.shape[0]) if (Y[k] == 1)]:
            n += 1
            if Prediction[i] == 1:
                tp += 1
        return [tp, n]

    def TPR(self, **kwargs):
        return self.TP(**kwargs)[0]/self.TP(**kwargs)[1]

    def FP(self, X, Y, hyper_para = [1e-4, 1e-2, 0.5], method = "log_cla", **kwargs):
        kwargs["X"] = X
        kwargs["Y"] = Y
        if "C" in kwargs:
            C = kwargs["C"]
        else:
            C = hyper_para[2]
        if method == "log_cla":
            Prediction = predict(**kwargs)
        else:
            Prediction = dec_boundary(predict(**kwargs), C)
        n = 0
        fp = 0
        for i in [k for k in range(X.shape[0]) if (Y[k] == 0)]:
            n += 1
            if Prediction[i] == 1:
                fp += 1
        return [fp + 1e-8, n]

    def FPR(self, **kwargs):
        return self.FP(**kwargs)[0]/self.FP(**kwargs)[1]

    def TN(self, X, Y, hyper_para = [1e-4, 1e-2, 0.5], method = "log_cla", **kwargs):
        kwargs["X"] = X
        kwargs["Y"] = Y
        if "C" in kwargs:
            C = kwargs["C"]
        else:
            C = hyper_para[2]
        if method == "log_cla":
            Prediction = predict(**kwargs)
        else:
            Prediction = dec_boundary(predict(**kwargs), C)
        n = 0
        tn = 0
        for i in [k for k in range(X.shape[0]) if (Y[k] == 0)]:
            n += 1
            if Prediction[i] == 0:
                tn += 1
        return [tn, n]

    def TNR(self, **kwargs):
        return self.TN(**kwargs)[0]/self.TN(**kwargs)[1]

    def FN(self, X, Y, hyper_para = [1e-4, 1e-2, 0.5], method = "log_cla", **kwargs):
        kwargs["X"] = X
        kwargs["Y"] = Y
        if "C" in kwargs:
            C = kwargs["C"]
        else:
            C = hyper_para[2]
        if method == "log_cla":
            Prediction = predict(**kwargs)
        else:
            Prediction = dec_boundary(predict(**kwargs), C)
        n = 0
        fn = 0
        for i in [k for k in range(X.shape[0]) if (Y[k] == 1)]:
            n += 1
            if Prediction[i] == 0:
                fn += 1
        return [fn + 1e-8, n]

    def FNR(self, **kwargs):
        return self.FN(**kwargs)[0]/self.FN(**kwargs)[1]

    def print_confusion_matrix(self, **kwargs):
        print("Act.\Pred.\t1\t0")
        print("1\t\t{}\t{}".format(int(self.TP(**kwargs)[0]), int(self.FN(**kwargs)[0])))
        print("0\t\t{}\t{}".format(int(self.FP(**kwargs)[0]), int(self.TN(**kwargs)[0])))

    class log_cla:
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

        def beta_grad(X, Y, beta, hyper_para = [1e-4, 1e-2, 0.5], **kwargs):
            kwargs["X"] = X
            kwargs["Y"] = Y
            kwargs["beta"] = beta
            kwargs["hyper_para"] = hyper_para
            if "Lambda" in kwargs:
                Lambda = kwargs["Lambda"]
            else:
                Lambda = hyper_para[0]

            grad = np.zeros((X.shape[1], 1))

            for i in range(X.shape[0]):
                grad += 1/X.shape[0] * X[i,:].reshape((X.shape[1], 1)) * (1/ (1 + exp(-X[i,:] @ beta)) - Y[i])

            return grad

    class log_reg:
        "loss and gradient for logistic regression"

        def calculate_loss(X, Y, beta, **kwargs):
            kwargs["X"] = X
            kwargs["Y"] = Y
            kwargs["beta"] = beta
            n = X.shape[0]
            loss = 0
            for i in range(n):
                loss += 1/n * ( -Y[i] * log(predict(**kwargs)[i], 2) - (1-Y[i]) * log(1 -predict(**kwargs)[i], 2) )
            return loss

        def beta_grad(X, Y, beta, hyper_para = [1e-4, 1e-2, 0.5], **kwargs):
            kwargs["X"] = X
            kwargs["Y"] = Y
            kwargs["beta"] = beta
            if "Lambda" in kwargs:
                Lambda = kwargs["Lambda"]
            else:
                Lambda = hyper_para[0]

            grad = np.zeros((X.shape[1], 1))

            for i in range(X.shape[0]):
                grad += 1/X.shape[0] * X[i,:].reshape((X.shape[1], 1)) * (predict(**kwargs)[i] - Y[i])

            return grad

class gd(loss):
    "NADAM"
    def __init__(self, X, Y,  hyper_para = [1e-4, 1e-2, 0.5], batch_number = 10,
                    method = "log_cla", max_iter = 1e3, eps = 1e-6, memory_size = 50,
                    eta1 = 0.9, eta2 = 0.999, e = 1e-8, **kwargs):
        loss.__init__(self, method, **kwargs)
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
                    * self.los.beta_grad(self.X, self.Y, self.beta, [self.Lambda, self.alpha, self.C])
            mhat = self.m / (1-self.eta1**(i+1))
            self.v = self.eta2 * self.v + (1-self.eta2)\
                    * (self.los.beta_grad(self.X, self.Y, self.beta, [self.Lambda, self.alpha, self.C]))**2
            vhat = self.v / (1-self.eta2**(i+1))
            self.beta = self.beta - self.alpha / (np.sqrt(vhat) + self.e)\
                    * (self.eta1 * mhat + (1-self.eta1)\
                    * self.los.beta_grad(self.X, self.Y, self.beta, [self.Lambda, self.alpha, self.C])\
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
                                                            [self.Lambda, self.alpha, self.C])
            mhat = self.m / (1-self.eta1**(i+1))
            self.v = self.eta2 * self.v + (1-self.eta2)\
                        * (self.los.beta_grad(X_batch, Y_batch, self.beta,
                                                            [self.Lambda, self.alpha, self.C]))**2
            vhat = self.v / (1-self.eta2**(i+1))
            self.beta = self.beta - self.alpha / (np.sqrt(vhat) + self.e)\
                        * (self.eta1 * mhat + (1-self.eta1)\
                        * self.los.beta_grad(X_batch, Y_batch, self.beta,
                                                            [self.Lambda, self.alpha, self.C])\
                        / (1-self.eta1**(i+1)))
            memory[i % self.memory_size] = np.sum(np.abs(self.beta - x))

            if i > self.memory_size and np.sum(memory) < self.eps:
                break

            if (i+1) % (self.batch_number) == 0:
                random_index = np.random.choice(self.batch_number, self.batch_number, replace = False)

        return self.beta

def general_get_beta(method = "log_cla", stoch = False, **kwargs):

    kwargs["method"] = method
    GD = gd(**kwargs)
    if stoch == True:
        return GD.stoch_grad_descent()
    else:
        return GD.grad_descent()

def get_parameters(X, Y, k_0 = 1, hyper_para = [1e-4, 1e-2, 0.5], method = "log_cla", **kwargs):
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
        kwargs["Lambda_grid"] = hyper_para[0]
    hyperparameters = [kwargs["Lambda_grid"]]


    if "alpha_grid" not in kwargs:
        kwargs["alpha_grid"] = hyper_para[1]
    hyperparameters.append(kwargs["alpha_grid"])

    if "C_grid" not in kwargs:
        kwargs["C_grid"] = hyper_para[2]

    # adding default C value for training
    hyperparameters.append(0.5)

    hyper_number = len(hyperparameters)


    # create grid of the hyperparameters
    hyper_grid = np.array(np.meshgrid(*hyperparameters)).T.reshape(-1, hyper_number)


    # find the best hyperparameters

    if k_0 > 1:   # do CV over k_0 folds

        print("\nsearch for hyper_para...")
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

        # check which treshold yields best outcome
        zero_one_loss = np.zeros((k_0,1))
        Zero_one_loss = 0
        best_treshold = 0
        print("\nsearch for treshold...")

        for i in range(len(kwargs["C_grid"])):
            # choosing treshold with lowest zero_one_loss
            kwargs["C"] = kwargs["C_grid"][i]
            kwargs["hyper_para"][2] = kwargs["C_grid"][i]
            # do the CV
            for j in range(k_0):

                first_cut = j * int(n / k_0)
                second_cut = j * int(n / k_0) + int(n / k_0)

                kwargs["X"] = np.delete(X, slice(first_cut, second_cut),axis=0)
                kwargs["Y"] = np.delete(Y, slice(first_cut, second_cut),axis=0)

                kwargs["beta"] = general_get_beta(**kwargs)

                kwargs["X"] = X[first_cut:second_cut,:]
                kwargs["Y"] = Y[first_cut:second_cut]

                if method == "log_cla":
                    zero_one_loss[j] = np.sum(np.abs(predict(**kwargs) - Y[first_cut:second_cut]))
                else:
                    zero_one_loss[j] = np.sum(np.abs(dec_boundary(predict(**kwargs)) - Y[first_cut:second_cut], kwargs["C"]))

            new_Zero_one_loss = np.mean(zero_one_loss)

            # choosing treshold via highest TPR/FPR
            #kwargs["C"] = kwargs["C_grid"][i]
            #kwargs["hyper_para"][2] = kwargs["C_grid"][i]
            #new_performance = loss(**kwargs).TPR(**kwargs)/loss(**kwargs).FPR(**kwargs)
            #print("treshold: {} \t performance: {}".format(round(kwargs["C"],2), round(new_performance,2)))

            # choose treshold with smallest 0-1-loss and a maximum FNR of 0.15
            if (new_Zero_one_loss < Zero_one_loss and loss(**kwargs).FNR(**kwargs) < 0.15) or i == 0:
                Zero_one_loss = new_Zero_one_loss
                best_treshold = i

        hyper_para[2] = kwargs["C_grid"][best_treshold]

        kwargs["C"] = kwargs["C_grid"][best_treshold]
        kwargs["hyper_para"] = hyper_para

        kwargs["X"] = X
        kwargs["Y"] = Y
        beta = general_get_beta(**kwargs)


    else:   #no CV

        kwargs["X"] = X
        kwargs["Y"] = Y

        # calculate possible results with the points in the grid
        error = 0
        hyper_para = []
        print("\nsearch for best hyper parameter...")

        for i in range(0, len(hyper_grid)):
            kwargs["hyper_para"] = hyper_grid[i][:]
            print("\nhyper_para: {}".format(kwargs["hyper_para"]))

            kwargs["beta"] = general_get_beta(**kwargs)

            new_error = loss(**kwargs).los.calculate_loss(**kwargs)

            # check whether new grid-point is better, and replace the results
            # if it is true
            if new_error < error or i == 0:
                error = new_error
                hyper_para = hyper_grid[i][:]

        # check which treshold yields best outcome
        zero_one_loss = 0
        best_treshold = 0
        print("\nsearch for treshold...")

        for i in range(len(kwargs["C_grid"])):
            # choosing treshold with lowest zero_one_loss
            kwargs["C"] = kwargs["C_grid"][i]
            kwargs["hyper_para"][2] = kwargs["C_grid"][i]
            if method == "log_cla":
                new_loss = np.sum(np.abs(predict(**kwargs) - Y))
            else:
                new_loss = np.sum(np.abs(dec_boundary(predict(**kwargs), kwargs["C"]) - Y0))


            # choosing treshold via highest TPR/FPR
            #kwargs["C"] = kwargs["C_grid"][i]
            #kwargs["hyper_para"][2] = kwargs["C_grid"][i]
            #new_performance = loss(**kwargs).TPR(**kwargs)/loss(**kwargs).FPR(**kwargs)
            #print("treshold: {} \t performance: {}".format(round(kwargs["C"],2), round(new_performance,2)))

            # choose treshold with smallest 0-1-loss and a maximum FNR of 0.15
            if (new_loss < zero_one_loss and loss(**kwargs).FNR(**kwargs) < 0.15) or i == 0:
                zero_one_loss = new_loss
                best_treshold = i

        hyper_para[2] = kwargs["C_grid"][best_treshold]

        kwargs["C"] = kwargs["C_grid"][best_treshold]
        kwargs["hyper_para"] = hyper_para

        beta = general_get_beta(**kwargs)

    return beta, hyper_para
