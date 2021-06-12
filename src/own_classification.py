import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
from scipy import stats


def predict(X, beta, method = "log_cla", hyper_para = [1e-4, 1e-2, 0.5],
            **kwargs):
    """
    Parameters
    ----------
    X: array
        (n,k) array of explanatory variables including a column of ones for the
        intercept
    beta: vector
        parameters of model
    method: "method"
        method we use to train which also specifies the loss
    hyper_para[2]: real value
        threshold to use as decision boundary
    """
    kwargs["X"] = X
    kwargs["beta"] = beta
    if "C" in kwargs:
        C = kwargs["C"]
    else:
        C = hyper_para[2]
    kwargs["method"] = method

    if method == "log_cla":
        eta = np.zeros((X.shape[0], 1))
        f = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            eta[i] = 1 / (1 + exp(-X[i,:] @ beta))
            if eta[i] - C > 0:
                f[i] = 1
        return f, eta

    else:
        exit(str(method) + "method not implemented")


class loss:
    """
    Parameters
    ----------
    method: "method"
        method we use to train which also specifies the loss
    """
    def __init__(self, method = "log_cla", **kwargs):
        kwargs["method"] = method
        if method == "log_cla":
            self.los = self.log_cla
        else:
            exit(str(method) + "method not implemented")


    def TP(self, X, Y, hyper_para = [1e-4, 1e-2, 0.5], method = "log_cla", **kwargs):
        kwargs["X"] = X
        kwargs["Y"] = Y
        if "C" in kwargs:
            C = kwargs["C"]
        else:
            C = hyper_para[2]
        if method == "log_cla":
            Prediction = predict(**kwargs)[0]

        n = 0
        tp = 0
        for i in [k for k in range(X.shape[0]) if (Y[k] == 1)]:
            n += 1
            if Prediction[i] == 1:
                tp += 1
        return tp, n

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
            Prediction = predict(**kwargs)[0]

        n = 0
        fp = 0
        for i in [k for k in range(X.shape[0]) if (Y[k] == 0)]:
            n += 1
            if Prediction[i] == 1:
                fp += 1
        return fp + 1e-8, n

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
            Prediction = predict(**kwargs)[0]

        n = 0
        tn = 0
        for i in [k for k in range(X.shape[0]) if (Y[k] == 0)]:
            n += 1
            if Prediction[i] == 0:
                tn += 1
        return tn, n

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
            Prediction = predict(**kwargs)[0]

        n = 0
        fn = 0
        for i in [k for k in range(X.shape[0]) if (Y[k] == 1)]:
            n += 1
            if Prediction[i] == 0:
                fn += 1
        return fn + 1e-8, n

    def FNR(self, **kwargs):
        return self.FN(**kwargs)[0]/self.FN(**kwargs)[1]

    def ACC(self, **kwargs):
        return (self.TP(**kwargs)[0] + self.TN(**kwargs)[0])/ \
        (self.TP(**kwargs)[1] + self.TN(**kwargs)[1])

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
                loss += - 1/n * ( Y[i] * (X[i,:] @ beta) \
                                    - log(1 + exp(X[i,:] @ beta), 2) )
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
                grad += X[i,:].reshape((X.shape[1], 1)) \
                        * (1/ (1 + exp(-X[i,:] @ beta)) - Y[i]) \
                        + Lambda * beta
            grad = grad/X.shape[0]
            return grad


class gd(loss):
    """
    Parameters
    ----------
    X: array
        (n,k) array of explanatory variables including a column of ones for the
        intercept
    Y: vector (also np.array)
        vector of all the labels of the measurments of the training data
    hyper_para: list
        hyperparameters used in the learning rule:
            hyper_para[0]: Lambda for a possible penalization
            (needs to be included in the gradient)
            hyper_para[1]: alpha as a stepwidth in the updating rule
            hyper_para_[2]: C threshold as decision boundary
            (negative log likelhood is independent of C, use 0-1-loss to
            decide for best threshold)
    batch_number: 0< int < number of Observations
        defines how many minibatches should be generated in the stochastic
        gradient descent
    method: "method"
        method we use to train which also specifies the loss
    max_iter: 0< int
        maximum number of iterations in learning algorithm
    eps: small real number
        gives minimum distance in memory, otherwise break
    memory_size:
        gives number of steps to include into memory
    eta1: 0 < real number < 1
        memory of first momentum in NADAM
    eta2: 0 < real number < 1
        memory of second momentum in NADAM
    e: small pos. real number
        prevents division by zero in NADAM
    """
    def __init__(self, X, Y,  hyper_para = [1e-4, 1e-2, 0.5],
                batch_number = 10, method = "log_cla", max_iter = 1e3,
                eps = 1e-6, memory_size = 50, eta1 = 0.9, eta2 = 0.999,
                e = 1e-8, **kwargs):
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
                    * self.los.beta_grad(self.X, self.Y, self.beta,\
                    [self.Lambda, self.alpha, self.C])
            mhat = self.m / (1-self.eta1**(i+1))
            self.v = self.eta2 * self.v + (1-self.eta2)\
                    * (self.los.beta_grad(self.X, self.Y, self.beta,\
                    [self.Lambda, self.alpha, self.C]))**2
            vhat = self.v / (1-self.eta2**(i+1))
            self.beta = self.beta - self.alpha / (np.sqrt(vhat) + self.e)\
                    * (self.eta1 * mhat + (1-self.eta1)\
                    * self.los.beta_grad(self.X, self.Y, self.beta,\
                    [self.Lambda, self.alpha, self.C])\
                    / (1-self.eta1**(i+1)))
            memory[i % self.memory_size] = np.sum(np.abs(self.beta - x))

            if i > self.memory_size and np.sum(memory) < self.eps:
                break

# ordinary gradient descent as a backup check if NADAM works
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
    """
    Paramerers
    ----------
    method = "method"
        method we use to train which also specifies loss
    stoch = Boolean
        chooses between stochastic gradient descent as learning algorithm (True)
        or ordinary gradient descent (False)

    Returns
    -------
    gd.(stoch_)grad_descent()
    """

    kwargs["method"] = method
    GD = gd(**kwargs)
    if stoch == True:
        return GD.stoch_grad_descent()
    else:
        return GD.grad_descent()


def get_parameters(X, Y, k_0 = 1, hyper_para = [1e-4, 1e-2, 0.5],
                    method = "log_cla", **kwargs):
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
    method = "method":
        method we use to train which also specifies the loss

    Returns
    -------
    beta: array
        trained model parameters
    hyper_para: array
        best hyperparameters training algorithm
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

        # check which threshold yields best outcome
        zero_one_loss = np.zeros((k_0,1))
        Zero_one_loss = 0
        best_threshold = 0
        print("\nsearch for threshold...")
        kwargs["beta"] = general_get_beta(**kwargs)
        for i in range(len(kwargs["C_grid"])):
            # choosing threshold with lowest zero_one_loss
            kwargs["C"] = kwargs["C_grid"][i]
            kwargs["hyper_para"][2] = kwargs["C_grid"][i]
            # do the CV
            for j in range(k_0):

                first_cut = j * int(n / k_0)
                second_cut = j * int(n / k_0) + int(n / k_0)

                kwargs["X"] = np.delete(X, slice(first_cut, second_cut),axis=0)
                kwargs["Y"] = np.delete(Y, slice(first_cut, second_cut),axis=0)

                kwargs["X"] = X[first_cut:second_cut,:]
                kwargs["Y"] = Y[first_cut:second_cut]

                if method == "log_cla":
                    zero_one_loss[j] = np.sum(np.abs(predict(**kwargs)[0] - Y[first_cut:second_cut]))
                else:
                    zero_one_loss[j] = np.sum(np.abs(dec_boundary(predict(**kwargs)[0]) - Y[first_cut:second_cut], kwargs["C"]))

            new_Zero_one_loss = np.mean(zero_one_loss)

            # choosing threshold via highest TPR/FPR
            #kwargs["C"] = kwargs["C_grid"][i]
            #kwargs["hyper_para"][2] = kwargs["C_grid"][i]
            #new_performance = loss(**kwargs).TPR(**kwargs)/loss(**kwargs).FPR(**kwargs)
            #print("threshold: {} \t performance: {}".format(round(kwargs["C"],2), round(new_performance,2)))

            # choose threshold with smallest 0-1-loss and a maximum FNR of
                                                                    # FNR_max
            if (new_Zero_one_loss < Zero_one_loss and loss(**kwargs).FNR(**kwargs) < 1) or i == 0:
                Zero_one_loss = new_Zero_one_loss
                best_threshold = i

        hyper_para[2] = kwargs["C_grid"][best_threshold]

        kwargs["C"] = kwargs["C_grid"][best_threshold]
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

        kwargs["hyper_para"] = hyper_para
        kwargs["beta"] = general_get_beta(**kwargs)
        beta = kwargs["beta"]

        # check which threshold yields best outcome
        zero_one_loss = 0
        best_threshold = 0
        print("\nsearch for threshold...")

        for i in range(len(kwargs["C_grid"])):
            # choosing threshold with lowest zero_one_loss
            kwargs["C"] = kwargs["C_grid"][i]
            kwargs["hyper_para"][2] = kwargs["C_grid"][i]
            if method == "log_cla":
                new_loss = np.sum(np.abs(predict(**kwargs)[0] - Y))
            else:
                new_loss = np.sum(np.abs(dec_boundary(predict(**kwargs)[0], kwargs["C"]) - Y))


            # choosing threshold via highest TPR/FPR
            #kwargs["C"] = kwargs["C_grid"][i]
            #kwargs["hyper_para"][2] = kwargs["C_grid"][i]
            #new_performance = loss(**kwargs).TPR(**kwargs)/loss(**kwargs).FPR(**kwargs)
            #print("threshold: {} \t performance: {}".format(round(kwargs["C"],2), round(new_performance,2)))

            # choose threshold with smallest 0-1-loss and a maximum FNR of
                                                                        #FNR_max
            if (new_loss < zero_one_loss and loss(**kwargs).FNR(**kwargs) < 1) or i == 0:
                zero_one_loss = new_loss
                best_threshold = i

        hyper_para[2] = kwargs["C_grid"][best_threshold]

        kwargs["C"] = kwargs["C_grid"][best_threshold]
        kwargs["hyper_para"] = hyper_para



    return beta, hyper_para

class SSM:
    def __init__(self, p_pred, Y, n = 50, **kwargs):
        self.p_pred = p_pred
        self.Y = Y
        self.n = n

    def SSM(self):

        ind_sort = np.argsort(self.p_pred, axis = 0)

        p_pred_sort = np.zeros(self.p_pred.shape)
        Y_sort = np.zeros(self.Y.shape)

        #sort p_pred and Y according to prob. in p_pred
        for i in range(len(ind_sort)):
            p_pred_sort[i,0] = self.p_pred[ind_sort[i],0]
            Y_sort[i,0] = self.Y[ind_sort[i],0]

        p_est = np.zeros(self.p_pred.shape)

        # estimate real probability via SSM
        for i in range(len(p_est)):
            if i < self.n:
                for j in range(-i, self.n):
                    p_est[i] += 1/(self.n + i) * Y_sort[(i+j)%Y_sort.shape[0]]
            elif i > (len(p_est) - self.n):
                for j in range(-self.n, (len(p_est) - i)):
                    p_est[i] += 1/(self.n + (len(p_est) - i)) *\
                                    Y_sort[(i+j)%Y_sort.shape[0]]
            else:
                for j in range(-self.n, self.n+1):
                    p_est[i] += 1/(2*self.n + 1) * Y_sort[(i+j)%Y_sort.shape[0]]

        return p_pred_sort, p_est

    def plot_prob_vs_prob(self):

        fig, ax = plt.subplots(figsize=(10,10))

        # ax.set_title("Actual and predicted probability via SSM", fontsize = 24)
        ax.set_xlabel("Predicted probability", fontsize = 23)
        ax.set_ylabel("Actual probability", fontsize = 23)

        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_yticks(np.arange(0,1.1,0.1))
        ax.tick_params(labelsize = 18)
        plt.grid()

        p_pred, p_est = self.SSM()
        slope, intercept, r_value, p_value, std_err = stats.linregress(p_pred[:,0], p_est[:,0])
        x = np.linspace(np.min(p_pred), np.max(p_pred), 1000)
        y = slope * x + intercept

        plt.plot(p_pred, p_est, "o", c = "r", alpha = 0.5)
        plt.plot(x, y, "b-", label = r"$y = {}\cdot x + {}$, $R^2 = {}$".format(round(slope,2), round(intercept,2), round(r_value**2,3)))

        plt.legend(loc = "best", fontsize = 23)
        fig.tight_layout(pad=1)
        plt.savefig("ProbabilitySSM.png")
        return 1
