import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ReLU, Dense, Dropout, ActivityRegularization
import numpy as np
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
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
import sys
import shutil
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
import random

def custom_loss_kl_pi_pitilde(weights, y_pred):
    global u_global
    return u_global * y_pred + tf.keras.backend.binary_crossentropy(weights, y_pred)

def create_data_loader(x_train, y_train, batch_size=32, shuffle=True):
    # Convert numpy arrays or lists to TensorFlow tensors
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train_Y_tensor = tf.convert_to_tensor(y_train[0], dtype=tf.float32)
    y_train_E_tensor = tf.convert_to_tensor(y_train[1], dtype=tf.float32)

    # Combine Y and E labels into a single tensor
    y_train_combined_tensor = tf.stack([y_train_Y_tensor, y_train_E_tensor], axis=1)

    # Create a TensorFlow Dataset with x_train and combined Y-E labels
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_tensor, y_train_combined_tensor))

    # Shuffle and batch the dataset
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=len(x_train))
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset

def negative_log_likelihood(E):
    def loss(y_true, y_pred):
        if tf.is_tensor(y_true):
            # If y_true is a TensorFlow tensor, convert it to a NumPy array
            sort_idx = tf.constant(np.argsort(y_true)[::-1], dtype=tf.int32)
            y_true, y_pred = tf.gather(y_true, sort_idx), tf.gather(y_pred, sort_idx)
        else:
            # If y_true is already a NumPy array, use it directly
            sort_idx = np.argsort(y_true)[::-1]
            y_true, y_pred = y_true[sort_idx], y_pred[sort_idx]
        
        
        # hazard_ratio = tf.math.exp(y_pred)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        hazard_ratio = tf.math.exp(y_pred)
        log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
        uncensored_likelihood = tf.transpose(y_pred) - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood_ = -tf.math.reduce_sum(censored_likelihood)    
        neg_likelihood = neg_likelihood_ / len(E)            
        return neg_likelihood   
    return loss


def evaluate(model, X, y, weights=None, batch_size=16, eps=1e-5):
    Y, E = y

    if len(X.shape) <=3:
        # If the data is 2D, no batching is needed
        if weights is None:
            output = np.squeeze(model.predict(X)) + eps
        else:
            output = np.squeeze(weights) + eps

        y_pred = np.log(output)
        loss_function = negative_log_likelihood(E)
        loss = loss_function(Y, y_pred)
        return loss
    else:
        # If the data is 3D or 4D, calculate loss over all samples with batching
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        all_y_pred = np.zeros_like(Y)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            batch_X = X[start_idx:end_idx]
            batch_Y = Y[start_idx:end_idx]

            if weights is None:
                output = np.squeeze(model.predict(batch_X)) + eps
            else:
                batch_weights = weights[start_idx:end_idx]
                output = np.squeeze(batch_weights) + eps

            batch_y_pred = np.log(output)
            all_y_pred[start_idx:end_idx] = batch_y_pred

        loss_function = negative_log_likelihood(E)
        loss = loss_function(Y, all_y_pred)
        return loss


def statdist(generator, pepochs=200, method="power", v_init=None, ptol = 0.00001):
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
        left = lu[:-1,:-1]
        right = -lu[:-1,-1]
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
            v_init = np.random.rand(n,)
        # mat = generator+eye is row stochastic, i.e. rows add up to 1.
        # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
        eps = 1.0 / np.max(np.abs(generator))
        mat = np.eye(n) + eps*generator
        A = mat.T
        # Find the leading left eigenvector, corresponding to eigenvalue 1
        _, vecs = spsl.eigs(A, k=1, v0=v_init)
        res = np.real(vecs[:,0])
        return (1.0 / res.sum()) * res
    if method == "power":
        '''
        Power iteration has linear convergence rate and slow for lambda2~lambda1. 
        But guarantees positive eigenvector, if started accordingly.
        '''
        if v_init is None:
            v = np.random.rand(n,)
        else:
            v = v_init
        # mat = generator+eye is row stochastic, i.e. rows add up to 1.
        # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
        eps = 1.0 / np.max(np.abs(generator))
        mat = np.eye(n) + eps * generator
        A = mat.T
        # Find the leading left eigenvector, corresponding to eigenvalue 1
        # normAest = np.sqrt(np.linalg.norm(A, ord=1) * np.linalg.norm(A, ord=np.inf))
        v = v/np.linalg.norm(v)
        v = np.dot(A,v)
        for ind_iter in range(pepochs):
            # v = Av/np.linalg.norm(Av)
            pre_v = v
            Av = np.dot(A,v)
            # lamda = np.dot(v.T, Av)
            # r = Av-v*lamda
            v = Av/np.linalg.norm(Av)
            r = v-pre_v
            normr = np.linalg.norm(r)
            if normr < ptol:
                break
        res = np.real(v)
        # print(f"Power method converged in {ind_iter} epochs")
        return (1.0 / res.sum()) * res
    else:
        raise RuntimeError("not (yet?) implemented")
        
        
def ilsrx(y_train, rho, weights, x_beta_b, u, monitor_loss, ranking, classed=False, inner_params=[50, 1e-4], eps=1e-5):
    """modified spectral ranking algorithm for partial ranking data. Remove the inner loop for top-1 ranking.
    n: number of items
    rho: penalty parameter
    sigmas is the additional term compared to ILSR
    x_beta_b is model(X) AKA tildepi
    """
    iepochs=inner_params[0]
    rtol=inner_params[1]
    
    Y_train, E_train = y_train
    if classed:
        reweight_matrix = get_reweights(x_train, Y_train, E_train, C_train, subgroups, weights)
    n = len(weights)
    ilsr_conv = False
    iter = 0
    epsilon=0.0001
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
        # for ranking in self.rankings: one ranking in survival analysis
        for i, winner in enumerate(ranking):
            if classed:    
                winner_weight = reweight_matrix[i,i]
            sum_weights = sum(weights[x] for x in ranking[i:]) + eps
            for loser in ranking[i + 1:]:
                if classed:
                    loser_weight = reweight_matrix[i:,i]
                    val = winner_weight / np.dot(loser_weight, np.array([weights[x] for x in ranking[i:]]))
                else:
                    val = 1/sum_weights
                chain[loser, winner] += val
        # each row sums up to 0
        chain -= np.diag(chain.sum(axis=1))
        weights_prev = np.copy(weights)
        weights = statdist(chain, v_init=weights)
        #ILSRX loss
        ilsrx_pred = np.log(np.squeeze(weights)+1e-5)
        loss_function = negative_log_likelihood(E_train)
        ilsrx_loss = loss_function(Y_train, ilsrx_pred)
        monitor_loss["ilsrx_loss"].append(ilsrx_loss)
        # Check convergence
        iter += 1
        ilsr_conv = np.linalg.norm(weights_prev - weights) < rtol * np.linalg.norm(weights) or iter >= iepochs
    return weights, chain, monitor_loss


def deepspectral(model, x_train, x_val, x_test, y_train, y_val, y_test, rho, monitor_loss, ranking, weights=None, x_beta_b=None, u=None, classed=False, gamma=1, epoch=0, inner_params=[50, 1e-4], avg_window=5, eps=1e-5):
    depochs=inner_params[0]
    rtol=inner_params[1]
    ilrsx_time = 0
    nn_time = 0
    u_time = 0
    weights_prev = np.copy(weights)
    
    ## pi update: no matter what the initial weights are, should come first.
    start = time()
    weights, chain, monitor_loss = ilsrx(y_train, rho, weights, x_beta_b, u, monitor_loss, ranking, classed=classed, inner_params=inner_params)
    end = time()
    ilrsx_time = (end - start)

    MSE_score = np.sum((x_beta_b-weights)**2)
    global u_global
    params_conv = False
    iter = 0
    window_losses=[100000]

    while not params_conv:
        
        if len(x_train.shape)>3:
            batch_size = 16
        else:
            batch_size = len(x_train)
        start = time()
        history = model.fit(x_train, weights, batch_size=batch_size, epochs=1, shuffle=False, verbose=0)
        nn_time += (end - start)  # log time
        # print("train loss", history.history['loss'][-1])
        window_losses.append(history.history['loss'][-1])
        new_pred = batched_predict(model, x_train, batch_size=batch_size, predict_function=True)
        # print(weights[:10].shape, model.predict(x_train[:10]).shape)
        x_beta_b = np.squeeze(new_pred)  # predict new scores

        iter += 1  # log number of epochs
        if iter <= avg_window:
            avg_loss=np.mean(window_losses[:-1])
        else:
            avg_loss = np.mean(window_losses[-avg_window - 1:-1])
        params_conv = (np.abs(avg_loss - window_losses[-1]) < 10*rtol*avg_loss) or iter >= depochs  # check conv.  
    ## dual update
    start = time()
    u += gamma * rho*(x_beta_b - weights) #reversed
    end = time()
    u_time = (end - start)
        
    u_global = np.copy(u)

    ilsrx_loss = evaluate(model, x_train, y_train, weights)
    val_loss = evaluate(model, x_val, y_val)
    train_loss = evaluate(model, x_train, y_train)

    test_AUC, test_IBS, test_CI, test_surv_MSE = sksurv_metrics(model, x_train, y_train, x_test, y_test, use_exp=False)
    val_AUC, val_IBS, val_CI, val_surv_MSE  = sksurv_metrics(model, x_train, y_train, x_val, y_val, use_exp=False)
    train_AUC, train_IBS, train_CI, train_surv_MSE  = sksurv_metrics(model, x_train, y_train, x_train, y_train, use_exp=False)

    print(f"Test Set Metrics:\nAUC: {test_AUC:.3f}\nIBS: {test_IBS:.3f}\nCI: {test_CI:.3f}\nSurvival MSE: {test_surv_MSE:.3f}\n")
    print(f"Validation Set Metrics:\nAUC: {val_AUC:.3f}\nIBS: {val_IBS:.3f}\nCI: {val_CI:.3f}\nSurvival MSE: {val_surv_MSE:.3f}\n")
    print(f"Training Set Metrics:\nAUC: {train_AUC:.3f}\nIBS: {train_IBS:.3f}\nCI: {train_CI:.3f}\nSurvival MSE: {train_surv_MSE:.3f}\n")

    monitor_loss["ilsrx_loss"].append(ilsrx_loss)
    monitor_loss["val_loss"].append(val_loss)
    monitor_loss["train_loss"].append(train_loss)
    monitor_loss["MSE_score"].append(MSE_score)
    monitor_loss["NNtrain_loss"].append(history.history['loss'][-1])
    times = {"ilrsx_time": ilrsx_time,
            "nn_time": nn_time,
            "u_time": u_time}
    return model, weights_prev, x_beta_b, u, monitor_loss, chain, times


def deepspectral_batch(model, x_train, x_train_full, x_val, x_test, y_train, y_train_full, y_val, y_test, rho, monitor_loss, ranking, weights=None, x_beta_b=None, u=None, classed=False, gamma=1, epoch=0, inner_params=[50, 1e-4], avg_window=5, eps=1e-5):
    depochs=inner_params[0]
    rtol=inner_params[1]
    ## pi update: no matter what the initial weights are, should come first.
    weights, chain, monitor_loss = ilsrx(y_train, rho, weights, x_beta_b, u, monitor_loss, ranking, classed=classed, inner_params=inner_params)

    MSE_score = np.sum((x_beta_b-weights)**2)
    global u_global
    params_conv = False
    iter = 0
    window_losses=[100000]

    while not params_conv:
        
        if len(x_train.shape)>3:
            batch_size = 16
        else:
            batch_size = len(x_train)

        history = model.fit(x_train, weights, batch_size=batch_size, epochs=1, shuffle=False, verbose=0)
        # print("train loss", history.history['loss'][-1])
        window_losses.append(history.history['loss'][-1])
        new_pred = batched_predict(model, x_train, batch_size=batch_size, predict_function=True)
        # print(weights[:10].shape, model.predict(x_train[:10]).shape)
        x_beta_b = np.squeeze(new_pred)  # predict new scores

        iter += 1  # log number of epochs
        if iter <= avg_window:
            avg_loss=np.mean(window_losses[:-1])
        else:
            avg_loss = np.mean(window_losses[-avg_window - 1:-1])
        params_conv = (np.abs(avg_loss - window_losses[-1]) < 10*rtol*avg_loss) or iter >= depochs  # check conv.  
    ## dual update
    u += gamma * rho*(x_beta_b - weights) #reversed
    u_global = np.copy(u)

    ilsrx_loss = evaluate(model, x_train, y_train, weights)
    val_loss = evaluate(model, x_val, y_val)
    train_loss = evaluate(model, x_train, y_train)

    test_AUC, test_IBS, test_CI, test_surv_MSE = sksurv_metrics(model, x_train_full, y_train_full, x_test, y_test, use_exp=False)
    val_AUC, val_IBS, val_CI, val_surv_MSE  = sksurv_metrics(model, x_train_full, y_train_full, x_val, y_val, use_exp=False)
    train_AUC, train_IBS, train_CI, train_surv_MSE  = sksurv_metrics(model, x_train_full, y_train_full, x_train, y_train, use_exp=False)

    print(f"Test Set Metrics:\nAUC: {test_AUC:.3f}\nIBS: {test_IBS:.3f}\nCI: {test_CI:.3f}\nSurvival MSE: {test_surv_MSE:.3f}\n")
    print(f"Validation Set Metrics:\nAUC: {val_AUC:.3f}\nIBS: {val_IBS:.3f}\nCI: {val_CI:.3f}\nSurvival MSE: {val_surv_MSE:.3f}\n")
    print(f"Training Set Metrics:\nAUC: {train_AUC:.3f}\nIBS: {train_IBS:.3f}\nCI: {train_CI:.3f}\nSurvival MSE: {train_surv_MSE:.3f}\n")

    monitor_loss["ilsrx_loss"].append(ilsrx_loss)
    monitor_loss["val_loss"].append(val_loss)
    monitor_loss["train_loss"].append(train_loss)
    monitor_loss["MSE_score"].append(MSE_score)
    monitor_loss["NNtrain_loss"].append(history.history['loss'][-1])
    return model, weights, x_beta_b, u, monitor_loss, chain

def admm_kl(x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, epochs, model, batch_size, classed, inner_params, patience=5): 
    n = len(x_train)
    rho = 1
    if batch_size < n:
        train_loader = create_data_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    weights = np.ones(batch_size, dtype=float) / batch_size
    x_beta_b = np.copy(weights)
    u = 0
    eps = 1e-5
    gamma = 1
    
    chain_list, weight_list, log = [], [], []
    monitor_loss = {}
    monitor_loss["val_loss"], monitor_loss["train_loss"], monitor_loss["ilsrx_loss"], monitor_loss["MSE_score"], monitor_loss["NNtrain_loss"] = [], [], [], [], []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f'./experiment_results/model_checkpoint_{timestamp}'
    sum_times = {"ilrsx_time": 0, "nn_time": 0, "u_time": 0}
    
    val_loss = evaluate(model, x_val, y_val)
    monitor_loss["val_loss"].append(val_loss)
    current_loss = val_loss
    save_model(model, checkpoint_path)
    # pseudo_train loss
    train_loss = evaluate(model, x_train, y_train)
    monitor_loss["train_loss"].append(train_loss)
    current_loss = float('inf')
    no_improvement_count = 0

    for i in range(epochs):
        gamma /= (i + 1)
        if batch_size<n:
            for x_train_batch, y_train_batch in train_loader:
                y_batch_Y = y_train_batch[:, 0].numpy()  # Extract Y labels
                y_batch_E = y_train_batch[:, 1].numpy()  # Extract E labels
                x_train_batch = x_train_batch.numpy()
                y_train_batch = (y_batch_Y, y_batch_E)
                ranking = np.argsort(y_batch_Y)[::1]
                model, weights, x_beta_b, u, monitor_loss, chain, times =deepspectral_batch(model, x_train_batch, x_train, x_val, x_test, y_train_batch, y_train, y_val, y_test, rho, monitor_loss, ranking, weights, x_beta_b, u, classed, gamma, i, inner_params)
        else:
            ranking = np.argsort(y_train[0].squeeze())[::1]
            model, weights, x_beta_b, u, monitor_loss, chain\
                                   =deepspectral(model, x_train, x_val, x_test, y_train, y_val, y_test, rho, monitor_loss, ranking, weights, x_beta_b, u, classed, gamma, i, inner_params)
        chain_list.append(chain)
        weight_list.append(weights)
        for key in times.keys():
            sum_times[key] += times[key]
                
        # print(i, monitor_loss["val_loss"][-1])
        if monitor_loss["val_loss"][-1] < current_loss:
            save_model(model, checkpoint_path)
            current_loss = monitor_loss["val_loss"][-1]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
    
        if no_improvement_count >= patience:
            print(f'Early stopping at epoch {i} as validation loss did not improve for {patience} consecutive epochs.')
            break
            
    
    model = load_model(checkpoint_path, compile=False)
    if classed:
        train_subindex = [np.argwhere(C_train == a) for a in subgroups]
        weights_share = []
        for i in range(len(subgroups)):
            weights_share.append(sum(weights[train_subindex[i]]))
        test_AUC, test_IBS, test_CI = DSL_metrics(model, weights_share, x_train, y_train, x_test, y_test)
        val_AUC, val_IBS, val_CI = DSL_metrics(model, weights_share, x_train, y_train, x_val, y_val)
    else:
        test_AUC, test_IBS, test_CI, test_surv_MSE = sksurv_metrics(model, x_train, y_train, x_test, y_test, use_exp=False, plot=True, model_name = "Spectral")
        val_AUC, val_IBS, val_CI, val_surv_MSE = sksurv_metrics(model, x_train, y_train, x_val, y_val, use_exp=False)
        print(test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE)
    try:
        shutil.rmtree(checkpoint_path)
        print(f"Successfully removed directory: {checkpoint_path}")
    except OSError as e:
        print(f"Error: {checkpoint_path} : {e.strerror}")

    log = [monitor_loss["NNtrain_loss"], monitor_loss["val_loss"], monitor_loss["train_loss"], monitor_loss["ilsrx_loss"], monitor_loss["MSE_score"], chain_list, weight_list]
    
    return test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, sum_times, log


def run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params=[100, 1e-4], l2_reg=16, classed=False):
    activation = 'relu'

    if len(x_train.shape) <3:
        n_features = x_train.shape[1]
        if ResNet:
            x_train = np.expand_dims(x_train, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)
            model = models.ResNet1D_tf(l2_reg=l2_reg) #default ResNet18
        else:
            model = models.MLP1D_tf(dims[0], activation, n_features, depth, dropout, l2_reg)
    else:
        if ResNet:
            x_train = np.expand_dims(x_train, axis=4)
            x_val = np.expand_dims(x_val, axis=4)
            x_test = np.expand_dims(x_test, axis=4)
            model = models.ResNet3D_tf(depth, l2_reg=l2_reg)
        else:
            model = models.Conv3D_tf(depth=depth)
    lr_decay = 1e-4
    optimizer = Adam(learning_rate=learning_rate, decay=lr_decay)
    model.compile(loss=custom_loss_kl_pi_pitilde, optimizer=optimizer)

    global u_global
    u_global = np.zeros(len(x_train), dtype=float)

    test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, sum_times, log = admm_kl(x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, epochs, model, batch_size, classed, inner_params)

    return test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, sum_times, log

def run_dsl(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params=[100, 1e-4], l2_reg=16, classed=False):
    return run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params, l2_reg, classed)

def run_cdsl(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params=[100, 1e-4], l2_reg=16):
    return run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params, l2_reg, classed=True)


class spectral(tf.keras.Model):
    def __init__(self, net, loss=None, optimizer=None, device=None):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        
    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=None):
        """Fit  model with inputs and targets. Where 'input' is the covariates, and
        'target' is a tuple with (durations, events).
        
        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.
            target {np.array, tensor or tuple} -- Target [durations, events]. 
        
        Keyword Arguments:
            batch_size {int} -- Elements in each batch (default: {256})
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
            shuffle {bool} -- If we should shuffle the order of the dataset (default: {True})
    
        Returns:
            TrainingLogger -- Training log
        """
    
    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        """Predict the survival function for `input`.
        See `prediction_surv_df` to return a DataFrame instead.

        Arguments:
            input {dataloader, tuple, np.ndarray, or torch.tensor} -- Input to net.

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            [TupleTree, np.ndarray or tensor] -- Predictions
        """
        raise NotImplementedError
        
    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                       num_workers=0):
        """Predict the hazard function for `input`.

        Arguments:
            input {dataloader, tuple, np.ndarray, or torch.tensor} -- Input to net.

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            grads {bool} -- If gradients should be computed (default: {False})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            [np.ndarray or tensor] -- Predicted hazards
        """
        raise NotImplementedError

        
    def compute_baseline_hazards(self, input, df_target, max_duration, batch_size, eval_=True, num_workers=0):

        return (df_target
                .assign(expg=np.exp(self.model(input, batch_size, True, eval_, num_workers=num_workers)))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col]/x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))

    def predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        max_duration = np.inf if max_duration is None else max_duration
        if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False, 
                                                           baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(self.predict(input, batch_size, True, eval_, num_workers=num_workers)).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), 
                            index=bch.index)