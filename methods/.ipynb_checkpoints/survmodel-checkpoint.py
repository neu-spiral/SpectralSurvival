import numpy as np
import pandas as pd
from methods.utils import *
from methods.metrics import *
import scipy.sparse.linalg as spsl
import scipy.linalg as spl
from tensorflow.keras.optimizers import Nadam,Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import os
from methods import models
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import ReLU, Dense, Dropout, ActivityRegularization
import sys
import shutil


def custom_loss_kl_pi_pitilde(weights, y_pred):
    global u_global
    return u_global * y_pred + tf.keras.backend.binary_crossentropy(weights, y_pred)


def negative_log_likelihood(E):
    def loss(y_true, y_pred):
        sort_idx = np.argsort(y_true)[::-1]
        y_true, y_pred = y_true[sort_idx], y_pred[sort_idx]
        # hazard_ratio = tf.math.exp(y_pred)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        hazard_ratio = tf.math.exp(y_pred)
        log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
        uncensored_likelihood = tf.transpose(y_pred) - log_risk
        censored_likelihood = uncensored_likelihood * E[sort_idx]
        neg_likelihood_ = -tf.math.reduce_sum(censored_likelihood)
        neg_likelihood = neg_likelihood_ / len(E)
        return neg_likelihood

    return loss

def JNPLL(hazards, times, events, journey_ids):
    def NPLL(E):
        def loss(y_true, y_pred):
            # Ensure inputs are tensors
    #         y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    #         y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

            # Sort y_true and y_pred based on y_true values in descending order
            sort_idx = tf.argsort(y_true, direction='DESCENDING')
            y_true = tf.gather(y_true, sort_idx)
            y_pred = tf.gather(y_pred, sort_idx)

            hazard_ratio = tf.math.exp(y_pred)
            log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
            uncensored_likelihood = y_pred - log_risk
            censored_likelihood = tf.cast(uncensored_likelihood, dtype=tf.float64) * tf.cast(tf.gather(E, sort_idx), dtype=tf.float64)
            neg_likelihood_ = -tf.reduce_sum(censored_likelihood)
            neg_likelihood = neg_likelihood_ / len(E)

            return neg_likelihood
        return loss
    
    unique_journeys = tf.unique(journey_ids)[0]
    total_loss = 0.0
    
    for journey in unique_journeys:
        mask = journey_ids == journey
        journey_hazards = tf.boolean_mask(hazards, mask)
        journey_times = tf.boolean_mask(times, mask)
        journey_events = tf.boolean_mask(events, mask)
        
        # Sort journey-specific data based on times in descending order
        sort_idx = tf.argsort(journey_times, direction='DESCENDING')
        sorted_hazards = tf.gather(journey_hazards, sort_idx)
        sorted_times = tf.gather(journey_times, sort_idx)
        sorted_events = tf.gather(journey_events, sort_idx)
        
        # Compute NLL for the sorted journey-specific data
        journey_loss = NPLL(sorted_events)(sorted_times, sorted_hazards)
        
        total_loss += journey_loss
    
    return total_loss


def breslow_estimator(y_train, risk_scores, eps=1e-5):
    """
    Compute the Breslow estimate of the baseline cumulative hazard.

    Parameters:
    - Y_train: array-like, observed times
    - E_train: array-like, event indicators (1 if event, 0 if censored)
    - risk_scores: array-like, risk scores for each observation

    Returns:
    - times: unique event times
    - H0: baseline cumulative hazard estimates at each unique event time
    """
    Y_train, E_train = y_train
    # Sort data by event times
    idx = np.argsort(Y_train)
    sorted_times = Y_train[idx]
    sorted_indicators = E_train[idx]
    sorted_risk_scores = risk_scores[idx]

    # Unique event times
    unique_times, counts = np.unique(sorted_times, return_counts=True)

    H0 = []
    running_risk_set = sorted_risk_scores  # not exp for DSL
    for t, count in zip(unique_times, counts):
        dN = sum(sorted_indicators[:count])
        denom = sum(running_risk_set)
        H0_t = dN / (denom + eps)
        H0.append(H0_t)

        # Update the running risk set
        running_risk_set = running_risk_set[count:]
        sorted_times = sorted_times[count:]
        sorted_indicators = sorted_indicators[count:]
    return unique_times, H0


class Spectral:
    def __init__(self, model, optimizer, rankings=None, batch_size=None, full_dataset=None):
        self.optimizer = optimizer
        self.model = model
        self.model.compile(loss=custom_loss_kl_pi_pitilde, optimizer=self.optimizer)
        self.batch_size = batch_size
        self.rankings = rankings
        self.full_dataset = full_dataset

    def evaluate(self, X, y, weights=None, eps=1e-5):
        if len(y)==2:
            Y, E = y
        elif len(y)==3:
            Y, E, J = y
            
        if self.batch_size is None:
            self.batch_size = len(X)
        if len(X.shape) <= 3:
            # If the data is 2D, no batching is needed
            if weights is None:
                output = np.squeeze(self.model.predict(X)) + eps
            else:
                output = np.squeeze(weights) + eps

            y_pred = np.log(output)
            if len(y)==2:
                loss_function = negative_log_likelihood(E)
                loss = loss_function(Y, y_pred)
            elif len(y)==3:
                loss_function = JNPLL
                loss = loss_function(y_pred, Y, E, J)
            return loss
        else:
            # If the data is 3D or 4D, calculate loss over all samples with batching
            num_samples = X.shape[0]
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size

            all_y_pred = np.zeros_like(Y)

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size

                batch_X = X[start_idx:end_idx]
                batch_Y = Y[start_idx:end_idx]

                if weights is None:
                    output = np.squeeze(self.model.predict(batch_X)) + eps
                else:
                    batch_weights = weights[start_idx:end_idx]
                    output = np.squeeze(batch_weights) + eps

                batch_y_pred = np.log(output)
                all_y_pred[start_idx:end_idx] = batch_y_pred

            if len(y)==2:
                loss_function = negative_log_likelihood(E)
                loss = loss_function(Y, all_y_pred)
            elif len(y)==3:
                loss_function = JNPLL
                loss = loss_function(all_y_pred, Y, E, J)
            return loss

    def statdist(self, generator, pepochs=200, method="power", v_init=None, ptol=0.00001):
        """Compute the stationary distribution of a Markov chain, described by its infinitesimal generator matrix.
        Computing the stationary distribution can be done with one of the following methods:
        - `kernel`: directly computes the left null space (co-kernel) the generator
          matrix using its LU-decomposition. Alternatively: ns = spl.null_space(generator.T)
        - `eigenval`: finds the leading left eigenvector of an equivalent
          discrete-time MC using `scipy.sparse.linalg.eigs`.
        - `power`: finds the leading left eigenvector of an equivalent
          discrete-time MC using power iterations. v_init is the initial eigenvector.
        """
        n = generator.shape[0]
        if method == "kernel":
            # `lu` contains U on the upper triangle, including the diagonal.
            lu, piv = spl.lu_factor(generator.T, check_finite=False)
            # The last row contains 0's only.
            left = lu[:-1, :-1]
            right = -lu[:-1, -1]
            # Solves system `left * x = right`. Assumes that `left` is
            # upper-triangular (ignores lower triangle.)
            res = spl.solve_triangular(left, right, check_finite=False)
            res = np.append(res, 1.0)
            return (1.0 / res.sum()) * res
        if method == "eigenval":
            '''
            Arnoldi iteration has cubic convergence rate, but does not guarantee positive eigenvector
            '''
            if v_init is None:
                v_init = np.random.rand(n, )
            # mat = generator+eye is row stochastic, i.e. rows add up to 1.
            # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
            eps = 1.0 / np.max(np.abs(generator))
            mat = np.eye(n) + eps * generator
            A = mat.T
            # Find the leading left eigenvector, corresponding to eigenvalue 1
            _, vecs = spsl.eigs(A, k=1, v0=v_init)
            res = np.real(vecs[:, 0])
            return (1.0 / res.sum()) * res
        if method == "power":
            '''
            Power iteration has linear convergence rate and slow for lambda2~lambda1. 
            But guarantees positive eigenvector, if started accordingly.
            '''
            if v_init is None:
                v = np.random.rand(n, )
            else:
                v = v_init
            # mat = generator+eye is row stochastic, i.e. rows add up to 1.
            # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
            eps = 1.0 / np.max(np.abs(generator))
            mat = np.eye(n) + eps * generator
            A = mat.T
            # Find the leading left eigenvector, corresponding to eigenvalue 1
            # normAest = np.sqrt(np.linalg.norm(A, ord=1) * np.linalg.norm(A, ord=np.inf))
            v = v / np.linalg.norm(v)
            v = np.dot(A, v)
            for ind_iter in range(pepochs):
                # v = Av/np.linalg.norm(Av)
                pre_v = v
                Av = np.dot(A, v)
                # lamda = np.dot(v.T, Av)
                # r = Av-v*lamda
                v = Av / np.linalg.norm(Av)
                r = v - pre_v
                normr = np.linalg.norm(r)
                if normr < ptol:
                    break
            res = np.real(v)
            # print(f"Power method converged in {ind_iter} epochs")
            return (1.0 / res.sum()) * res
        else:
            raise RuntimeError("not (yet?) implemented")

    def ilsrx(self, y_train, rho, weights, x_beta_b, u, monitor_loss, inner_params=[100, 1e-4],
              eps=1e-5):
        """modified spectral ranking algorithm for partial ranking data. Remove the inner loop for top-1 ranking.
        n: number of items
        rho: penalty parameter
        sigmas is the additional term compared to ILSR
        x_beta_b is model(X) AKA tildepi
        """
        iepochs = inner_params[0]
        rtol = inner_params[1]
        if len(y_train)==2:
            Y_train, E_train = y_train
        elif len(y_train)==3:
            Y_train, E_train, J_train = y_train

        n = len(weights)
        ilsr_conv = False
        iter = 0
        epsilon = 0.0001
        while not ilsr_conv:
            # u is reversed
            sigmas = rho * (1 + np.log(np.divide(weights, x_beta_b + epsilon) + epsilon) - u)
            pi_sigmas = weights * sigmas
            # indices of states for which sigmas < 0
            ind_minus = np.where(sigmas < 0)[0]
            # indices of states for which sigmas >= 0
            ind_plus = np.where(sigmas >= 0)[0]
            # sum of pi_sigmas over states for which sigmas >= 0
            scaled_sigmas_plus = 2 * sigmas[ind_plus] / (np.sum(pi_sigmas[ind_minus]) - np.sum(pi_sigmas[ind_plus]))
            # fill up the transition matrix
            chain = np.zeros((n, n), dtype=float)
            # increase the outgoing rate from ind_plus to ind_minus
            for ind_minus_cur in ind_minus:
                chain[ind_plus, ind_minus_cur] = pi_sigmas[ind_minus_cur] * scaled_sigmas_plus

            for ranking in self.rankings: #one ranking in survival analysis
                for i, winner in enumerate(ranking):
                    sum_weights = sum(weights[x] for x in ranking[i:]) + eps
                    for loser in ranking[i + 1:]:
                        val = 1 / sum_weights
                        chain[loser, winner] += val
#             for ranking in self.rankings:
#                 sum_weights = sum(weights[x] for x in ranking) + epsilon
#                 for i, winner in enumerate(ranking):
#                     val = 1.0 / sum_weights
#                     for loser in ranking[i + 1:]:
#                         chain[loser, winner] += val
#                     sum_weights -= weights[winner]
            
#             else:
#                 # for ranking in self.rankings: #one ranking in survival analysis
#                 for i, winner in enumerate(ranking):
#                     sum_weights = sum(weights[x] for x in ranking[i:]) + eps
#                     for loser in ranking[i + 1:]:
#                         val = 1 / sum_weights
#                         chain[loser, winner] += val
            # each row sums up to 0
            chain -= np.diag(chain.sum(axis=1))
            weights_prev = np.copy(weights)
            weights = self.statdist(chain, v_init=weights)
            # ILSRX loss
#             ilsrx_pred = np.log(np.squeeze(weights) + 1e-5)
#             loss_function = negative_log_likelihood(E_train)
#             ilsrx_loss = loss_function(Y_train, ilsrx_pred)
#             monitor_loss["ilsrx_loss"].append(ilsrx_loss)
            # Check convergence
            iter += 1
            ilsr_conv = np.linalg.norm(weights_prev - weights) < rtol * np.linalg.norm(weights) or iter >= iepochs
        return weights, chain, monitor_loss

    def train_step_spectral(self, x_train, x_val, x_test, y_train, y_val, y_test, rho, monitor_loss, weights=None,
                     x_beta_b=None, u=None, gamma=1, inner_params=[100, 1e-4], avg_window=5,
                     eps=1e-5):
        depochs = inner_params[0]
        rtol = inner_params[1]
        global u_global
        params_conv = False
        iter = 0
        window_losses = [100000]
        nn_time = 0
        weights_prev = np.copy(weights)
        ## pi update: no matter what the initial weights are, should come first.
        start = time()
        weights, chain, monitor_loss = self.ilsrx(y_train, rho, weights, x_beta_b, u, monitor_loss,
                                             inner_params=inner_params)
        end = time()
        ilrsx_time = (end - start)

        MSE_score = np.sum((x_beta_b - weights) ** 2)

        while not params_conv:

            if len(x_train.shape) > 3:
                batch_size = 16
            else:
                if self.batch_size is None:
                    batch_size = len(x_train)
                else:
                    batch_size = self.batch_size
            start = time()
            history = self.model.fit(x_train, weights, batch_size=batch_size, epochs=1, shuffle=False, verbose=0)
            end = time()
            nn_time += (end - start)  # log time
            # print("train loss", history.history['loss'][-1])
            window_losses.append(history.history['loss'][-1])

            iter += 1  # log number of epochs
            if iter <= avg_window:
                avg_loss = np.mean(window_losses[:-1])
            else:
                avg_loss = np.mean(window_losses[-avg_window - 1:-1])
            params_conv = (np.abs(
                avg_loss - window_losses[-1]) < 10 * rtol * avg_loss) or iter >= depochs  # check conv.
        ## dual update
        start = time()
        x_beta_b = np.squeeze(self.model.predict(x_train))  # predict new scores
        end = time()
        predict_time = (end - start)

        start = time()
        u += gamma * rho * (x_beta_b - weights)  # reversed
        end = time()
        u_time = (end - start)

        u_global = np.copy(u)
        start = time()
        ilsrx_loss = self.evaluate(x_train, y_train, weights)
        val_loss = self.evaluate(x_val, y_val)
        train_loss = self.evaluate(x_train, y_train)
        end = time()
        eval_time = (end - start)
        if len(y_train)==2:
            test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(self.model, x_train, y_train, x_test, y_test, use_exp=False)
            val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(self.model, x_train, y_train, x_val, y_val, use_exp=False)
        else:
            # As spectral only takes unique samples
            test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(self.model, self.full_dataset[0], self.full_dataset[1], x_test, y_test, use_exp=False)
            val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(self.model, self.full_dataset[0], self.full_dataset[1], x_val, y_val, use_exp=False)
        
        print(test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE)

        monitor_loss["ilsrx_loss"].append(ilsrx_loss)
        monitor_loss["val_loss"].append(val_loss)
        monitor_loss["train_loss"].append(train_loss)
        monitor_loss["MSE_score"].append(MSE_score)
        monitor_loss["NNtrain_loss"].append(history.history['loss'][-1])
        times = {"ilrsx_time": ilrsx_time,
                 "nn_time": nn_time,
                 "u_time": u_time,
                 "eval_time": eval_time,
                 "predict_time": predict_time}
        return weights_prev, x_beta_b, u, monitor_loss, chain, times

    def admm_kl(self, x_train, x_val, x_test, y_train, y_val, y_test, epochs,
                inner_params, patience=10):
        n = len(x_train)
        rho = 1
        weights = np.ones(n, dtype=float) / n
        x_beta_b = np.copy(weights)
        u = 0
        eps = 1e-5
        gamma = 1
        chain_list, weight_list, log = [], [], []
        monitor_loss = {}
        monitor_loss["val_loss"], monitor_loss["train_loss"], monitor_loss["ilsrx_loss"], monitor_loss["MSE_score"], \
        monitor_loss["NNtrain_loss"] = [], [], [], [], []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f'./experiment_results/model_checkpoint_{timestamp}'
        sum_times = {"ilrsx_time": 0, "nn_time": 0, "predict_time": 0, "u_time": 0, "eval_time": 0, "spectral_time": 0, "save_time": 0}

        val_loss = self.evaluate(x_val, y_val)
        monitor_loss["val_loss"].append(val_loss)
        # save_model(self.model, checkpoint_path)
        # pseudo_train loss
        train_loss = self.evaluate(x_train, y_train)
        monitor_loss["train_loss"].append(train_loss)
        current_loss = float('inf')
        no_improvement_count = 0

        start = time()
        for i in range(epochs):
            gamma /= (i + 1)
            start_spectral = time()
            weights, x_beta_b, u, monitor_loss, chain, times \
                = self.train_step_spectral(x_train, x_val, x_test, y_train, y_val, y_test, rho, monitor_loss, 
                               weights, x_beta_b, u, gamma, inner_params)
            end_spectral = time()
            spectral_time = (end_spectral - start_spectral)
            sum_times["spectral_time"] += spectral_time

            chain_list.append(chain)
            weight_list.append(weights)
            start_save = time()
            for key in times.keys():
                sum_times[key] += times[key]

            if monitor_loss["val_loss"][-1] < current_loss:
                save_model(self.model, checkpoint_path)
                current_loss = monitor_loss["val_loss"][-1]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(
                    f'Early stopping at epoch {i} as validation loss did not improve for {patience} consecutive epochs.')
                break
            end_save = time()
            save_time = (end_save - start_save)
            sum_times["save_time"] += save_time

        end = time()
        admm_time = (end - start)
        sum_times["admm_time"] = admm_time

        start = time()
        self.model = load_model(checkpoint_path, compile=False)

        if len(y_train)==2:
            test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(self.model, x_train, y_train, x_test, y_test, use_exp=False)
            val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(self.model, x_train, y_train, x_val, y_val, use_exp=False)
        else:
            # As spectral only takes unique samples
            test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(self.model, self.full_dataset[0], self.full_dataset[1], x_test, y_test, use_exp=False)
            val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(self.model, self.full_dataset[0], self.full_dataset[1], x_val, y_val, use_exp=False)
            
        print(test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE)
        try:
            shutil.rmtree(checkpoint_path)
            print(f"Successfully removed directory: {checkpoint_path}")
        except OSError as e:
            print(f"Error: {checkpoint_path} : {e.strerror}")
        end = time()
        metric_time = (end - start)
        sum_times["metric_time"] = metric_time

        log = [monitor_loss["NNtrain_loss"], monitor_loss["val_loss"], monitor_loss["train_loss"],
               monitor_loss["ilsrx_loss"], monitor_loss["MSE_score"], chain_list, weight_list]

        return test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, sum_times, log


    def predict_surv(self, x_train, y_train, x_test, y_test):
        X_cumhazard = self.predict_cumulative_hazards(x_train, y_train, x_test, y_test)
        survival_matrix = np.exp(-X_cumhazard.astype(float))
        return survival_matrix

    def predict_hazard(self, x):
        if self.batch_size is None:
            batch_size = len(x_train)
        else:
            batch_size = self.batch_size
        risk_scores = batched_predict(self.model, x, batch_size=batch_size)
        return risk_scores

    def compute_baseline_hazards(self, x_train, y_train):
        risk_scores = self.predict_hazard(x_train)
        unique_times, H0 = self.breslow_estimator(y_train, risk_scores)
        return unique_times, H0

    def predict_cumulative_hazards(self, x_train, y_train, x_test, y_test):
        Y_test, E_test = y_test
        relative_hazard = self.predict_hazard(x_test)
        unique_times, H0 = self.compute_baseline_hazards(x_train, y_train)

        H0_cumulative = np.cumsum(H0)
        start, end = Y_test.min(), Y_test.max()
        bandwidth = (end - start) / n_band_split  # Adjust based on your dataset
        time_grid = np.linspace(start + eps, end - eps, num=100)

        smoothed_H0 = [kernel_smoothed_hazard(t, unique_times, H0_cumulative, bandwidth) for t in time_grid]
        smoothed_H0 = np.array(smoothed_H0)

        # Calculate the cumulative hazard for the test set
        X_cumhazard = np.matmul(relative_hazard.reshape(-1, 1), smoothed_H0.reshape(1, -1))
        return X_cumhazard


def run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, rankings, inner_params=[100, 1e-4], l2_reg=0, full_dataset=None):
    activation = 'relu'
    # get model by data
    if len(x_train.shape) <3:
        n_features = x_train.shape[1]
        if ResNet:
            x_train = np.expand_dims(x_train, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)
            model = models.ResNet1D_tf(l2_reg=2) #default ResNet18
        else:
            model = models.MLP1D_tf(dims[0], activation, n_features, depth, dropout, l2_reg)
    else:
        if ResNet:
            x_train = np.expand_dims(x_train, axis=4)
            x_val = np.expand_dims(x_val, axis=4)
            x_test = np.expand_dims(x_test, axis=4)
            model = models.ResNet3D_tf(depth, l2_reg=2)
        else:
            model = models.Conv3D_tf(depth=depth)

    optimizer = Adam(learning_rate=learning_rate)
    # define and train Spectral
    spectral = Spectral(model, optimizer, rankings, batch_size, full_dataset=full_dataset)
    global u_global
    u_global = np.zeros(len(x_train), dtype=float)

    test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, sum_times, log = spectral.admm_kl(x_train, x_val, x_test, y_train, y_val, y_test, epochs, inner_params)

    return test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, sum_times, log

def run_dsl(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, rankings, inner_params=[100, 1e-4], l2_reg=2, full_dataset=None):
    return run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, rankings, inner_params, l2_reg, full_dataset=full_dataset)
