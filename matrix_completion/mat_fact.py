import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from .utilis import test_train_split


class MF():
    
    def __init__(self, R, K, alpha, beta, iterations, lambda_bias, test=None):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        if isinstance(R, pd.DataFrame):
            R = R.values
        if test is not None and isinstance(test, pd.DataFrame):
            test = test.values
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.lambda_bias = lambda_bias
        self.iterations = iterations
        self.test = test
        self.best_it = np.infty
        self.best_mse = np.infty
    
    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        test_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            # if we have test data then we are training the model and we should store 
            # all these elements. If there is no test data then we are doing a prediction
            if self.test is not None:
                mse = self.mse()
                training_process.append((i, mse))
                temp_pred = self.full_matrix()
                test_error = self.get_mse(temp_pred, self.test)
                if test_error < self.best_mse:
                    self.best_mse = test_error
                    self.best_it = i+1
    #                 self.best_solution = Solution(self.b_u.copy(), self.b_i.copy(), self.P.copy(), self.Q.copy(), self.b)
                test_process.append(test_error)

        return training_process, test_process, self.best_mse, self.best_it

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        error /= len(xs)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.lambda_bias * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.lambda_bias * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)  + self.b
    
    def get_mse(self, pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)


class MFSLess():
        
    def train(self, R, K, alpha, beta, iterations, lambda_bias, test=None):
        """
        """
        if isinstance(R, pd.DataFrame):
                R = R.values
        if test is not None and isinstance(test, pd.DataFrame):
            test = test.values
        num_users, num_items = R.shape

        # store the bests results
        best_mse =  np.infty
        best_it = np.infty

        # Initialize user and item latent feature matrice
        P = np.random.normal(scale=1./K, size=(num_users, K))
        Q = np.random.normal(scale=1./K, size=(num_items, K))
        
        # Initialize the biases
        b_u = np.zeros(num_users)
        b_i = np.zeros(num_items)
        b = np.mean(R[np.where(R != 0)])
        
        # Create a list of training samples
        samples = [
            (i, j, R[i, j])
            for i in range(num_users)
            for j in range(num_items)
            if R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        test_process = []
        if test is not None:
            for i in range(iterations):
                np.random.shuffle(samples)
                self.sgd(samples, P, Q, alpha, beta, lambda_bias, b_u, b_i, b)
                # if we have test data then we are training the model and we should store 
                # all these elements. If there is no test data then we are doing a prediction
                mse = self.mse(R, b_u, b_i, P, Q, b)
                training_process.append((i, mse))
                temp_pred = self.full_matrix(b_u, b_i, P, Q, b)
                test_error = self.get_mse(temp_pred, test)
                if test_error < best_mse:
                    best_mse = test_error
                    best_it = i+1
    #                 self.best_solution = Solution(self.b_u.copy(), self.b_i.copy(), self.P.copy(), self.Q.copy(), self.b)
                test_process.append(test_error)
            return training_process, test_process, best_mse, best_it, K, alpha, beta, lambda_bias
        else:
            for i in range(iterations):
                np.random.shuffle(samples)
                self.sgd(samples, P, Q, alpha, beta, lambda_bias, b_u, b_i, b)
        return b_u, b_i, P, Q, b

    def mse(self, R, b_u, b_i, P, Q, b):
        """
        A function to compute the total mean square error
        """
        xs, ys = R.nonzero()
        predicted = self.full_matrix(b_u, b_i, P, Q, b)
        error = 0
        for x, y in zip(xs, ys):
            error += pow(R[x, y] - predicted[x, y], 2)
        error /= len(xs)
        return np.sqrt(error)

    def sgd(self, samples, P, Q, alpha, beta, lambda_bias, b_u, b_i, b):
        """
        Perform stochastic graident descent
        """
        for i, j, r in samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j, b, b_u, b_i, P, Q)
            e = (r - prediction)
            
            # Update biases
            b_u[i] += alpha * (e - lambda_bias * b_u[i])
            b_i[j] += alpha * (e - lambda_bias * b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = P[i, :][:]
            
            # Update user and item latent feature matrices
            P[i, :] += alpha * (e * Q[j, :] - beta * P[i,:])
            Q[j, :] += alpha * (e * P_i - beta * Q[j,:])

    def get_rating(self, i, j, b, b_u, b_i, P, Q):
        """
        Get the predicted rating of user i and item j
        """
        prediction = b + b_u[i] + b_i[j] + P[i, :].dot(Q[j, :].T)
        return prediction
    
    def full_matrix(self, b_u, b_i, P, Q, b):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return b_u[:,np.newaxis] + b_i[np.newaxis:,] + P.dot(Q.T)  + b
    
    def get_mse(self, pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)
    
    
class MatrixFactorization():
    
    def __init__(self, data):
        self.data = data
        self.best_param = dict()
        self.istrained = False
        self.global_best = dict()
    
    def grid_search(self, parameters, kfold=5, iter_max=100, paral=True, n_cores=-1):
        """ Does an exhaustive search of the grid and find the root mean square for each parameter in 
        cross validation manners. data is split using  random seed and the best parameters are found 
        using the test data for that split as validation measure. We keep track of that bese mse on the 
        test data as well as the curresponding iteration at which that result is found
       
        Arguments:
        -------------------------
            :parameters: dict: 
                    keys are reserved to these elements: 'alphas', 'betas', 'ks', 'lambdas'
                    values are lists of values that the key element may take
            :kfold: default=5:
                    number of splits that should be made for training on. It mimicks cross validation. 
                    However, unlike cross validation, this split of the training data does not reinforce that
                    the training datasets are disjoints. In addition the split is conducted on the wholde dataset 
                    not just the training dataset
            :iter_max: defaut=100:
                    maximum number of iterations used in stochastic gradient descent
        
        Returns:
        --------------------------
            :param_best: dict:
                    Dictionary of keys and best mean squared error and corresponding number of iterations 
                    found for each combination of the keys during the grid search.
                    
                        
        """
        seeds = np.random.choice(range(1000), size=kfold, replace=False)
        trains = []; tests = []
        for seed in seeds:
            test_, train_ = test_train_split(self.data, seed=seed)
            tests.append(test_.copy())
            trains.append(train_.copy())
        alphas = parameters['alphas']
        betas = parameters['betas']
        ks = parameters['ks']
        lambda_bias = parameters['lambdas']
        if paral: # we want to run the parallel version
            for i in range(len(trains)):
                mf = MFSLess()
                results = Parallel(n_jobs=n_cores)(delayed(mf.train)(trains[i], k, alpha, beta, iter_max, lam_bias, tests[i]) 
                            for k in ks for alpha in alphas for beta in betas for lam_bias in lambda_bias)
    
                # Loop through the results and store the with the keys
                for result in results:
                    _, _, best_mse, best_it, k, alpha, beta, lam_bias = result
                    if (k, alpha, beta, lam_bias) not in self.best_param:
                        self.best_param[(k, alpha, beta, lam_bias)] = dict()
                        self.best_param[(k, alpha, beta, lam_bias)]['mse'] = []
                        self.best_param[(k, alpha, beta, lam_bias)]['iter'] = []
                    self.best_param[(k, alpha, beta, lam_bias)]['mse'].append(best_mse)
                    self.best_param[(k, alpha, beta, lam_bias)]['iter'].append(best_it)
                
        else: # run serial version
            for i in range(len(trains)):
                for alpha in alphas:
                    for beta in betas:
                        for k in ks:
                            for lam_bias in lambda_bias:
                                # K, alpha, beta, iterations, lambda_bias, test=None
                                mf = MF(trains[i], k, alpha, beta, iter_max, lam_bias, tests[i])
                                _, _, best_mse, best_it = mf.train()
                                if (k, alpha, beta, lam_bias) not in self.best_param:
                                    self.best_param[(k, alpha, beta, lam_bias)] = dict()
                                    self.best_param[(k, alpha, beta, lam_bias)]['mse'] = []
                                    self.best_param[(k, alpha, beta, lam_bias)]['iter'] = []
                                self.best_param[(k, alpha, beta, lam_bias)]['mse'].append(best_mse)
                                self.best_param[(k, alpha, beta, lam_bias)]['iter'].append(best_it)

        self.istrained = True
        return self.best_param
    
    def fit(self):
        """
        A method to call once we have the best parameters
        It will check amount all the parameters from the grid search which one is the best
        Should retrun the full matrix with the bias, P and Q being the best found so far, the K
        """
        assert self.istrained, "Please train the model first using 'grid_search method'"
        for (k, alpha, beta, lam_bias) in self.best_param:
            self.best_param[(k, alpha, beta, lam_bias)]['mean'] = np.mean(self.best_param[(k, alpha, beta, lam_bias)]['mse'])
            self.best_param[(k, alpha, beta, lam_bias)]['sd'] = np.std(self.best_param[(k, alpha, beta, lam_bias)]['mse'])
            # if there is already  key then check if its values are worse than the one in best_parameters
            if len(self.global_best) > 0:
                for key in self.global_best:
                    val = self.global_best[key]['mean']
                    if val > self.best_param[(k, alpha, beta, lam_bias)]['mean'] or \
                        (val == self.best_param[(k, alpha, beta, lam_bias)]['mean'] and \
                         self.global_best[key]['sd'] > self.best_param[(k, alpha, beta, lam_bias)]['sd']):
                        # if that is the case then replace the current key by a new one and update the values
                        self.global_best.pop(key)
                        self.global_best[(k, alpha, beta, lam_bias)] = {'mean': self.best_param[(k, alpha, beta, lam_bias)]['mean'], 
                                                                'sd':self.best_param[(k, alpha, beta, lam_bias)]['sd'],
                                                                'iter': np.max(self.best_param[(k, alpha, beta, lam_bias)]['iter'])}                        
            else: # empty dictionary
                self.global_best[(k, alpha, beta, lam_bias)] = {'mean': self.best_param[(k, alpha, beta, lam_bias)]['mean'], 
                                                                'sd': self.best_param[(k, alpha, beta, lam_bias)]['sd'],
                                                               'iter': int(np.mean(self.best_param[(k, alpha, beta, lam_bias)]['iter']))}
                    

    def predict(self, pred_data):
        """ Predicts t
        """
        assert len(self.global_best) == 1, 'Call the method fit first' # we only have one key that is the best parameters
        for key in self.global_best:
             k, alpha, beta, lam_bias = key
        mf = MF(pred_data, k, alpha, beta, self.global_best[key]['iter'], lam_bias)
        mf.train()
        self.pred = mf.full_matrix()
        return self.pred

    def _abline(self):
        ""
        gca = plt.gca()
        gca.set_autoscale_on(False)
        gca.plot(gca.get_xlim(),gca.get_ylim(), 'red')
    
    def plot_predicted_actual(self, test):
        """
        """
        if isinstance(test, pd.DataFrame):
            test =  test.values
        
        pred = self.predict(test)
        x = test[test.nonzero()].flatten()
        y = pred[test.nonzero()].flatten()
        plt.scatter(x, y, alpha = 0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        self._abline()
        plt.show()

        model = LinearRegression()
        x_ = x.copy().reshape(-1, 1)
        model.fit(x_, y)
        print('coefficient of determination:', model.score(x_, y))
        print(f'y={model.intercept_} + {model.coef_} * x')