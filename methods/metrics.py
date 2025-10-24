import numpy as np
import os
import torch

import math
import pandas as pd

from lifelines import utils

from pycox.evaluation import EvalSurv
from sksurv.metrics import brier_score, integrated_brier_score, cumulative_dynamic_auc, concordance_index_censored
from lifelines import KaplanMeierFitter
from methods.utils import *
from methods.plot import *
from matplotlib import pyplot as plt

def surv_rmse(Y_test, E_test, survival_matrix, time_grid):
    test_sur = emp_surv(Y_test, E_test, time_grid, "test emp survival", line_style="-.")
    MSE = np.sqrt(np.mean((np.mean(survival_matrix, axis=0)-test_sur)**2))
    return MSE


def conditional_survival_for_censoring(T, E):
    """
    Compute the conditional survival function of censoring times using Kaplan-Meier.
    
    Parameters:
    - T: array-like, observed times (either event or censoring times) for each individual
    - E: array-like, event indicators (1 if event, 0 if censored)
    
    Returns:
    - kmf: KaplanMeierFitter object fitted for censoring times
    """
    
    # Filter out subjects who had an event
    censoring_times = T[E == 0]
    
    # Fit the Kaplan-Meier estimator
    kmf = KaplanMeierFitter()
    kmf.fit(censoring_times, event_observed=np.ones_like(censoring_times))
    
    return kmf


def get_CI(model, E_train, Y_train, x_train, E_val, Y_val, X_val, E_test, Y_test, x_test):
    Y_pred_train = -model.predict(x_train)
    train_CI = utils.concordance_index(Y_train, Y_pred_train, E_train)
    Y_pred_val = -model.predict(X_val)
    val_CI = utils.concordance_index(Y_val, Y_pred_val, E_val)
    Y_pred_test = -model.predict(x_test)
    test_CI = utils.concordance_index(Y_test, Y_pred_test, E_test)
    return train_CI, val_CI, test_CI 


def breslow_estimator(Y_train, E_train, risk_scores, eps=1e-5):
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
        
    # Sort data by event times
    idx = np.argsort(Y_train)
    sorted_times = Y_train[idx]
    sorted_indicators = E_train[idx]
    sorted_risk_scores = risk_scores[idx]

    # Unique event times
    unique_times, counts = np.unique(sorted_times, return_counts=True)
    
    H0 = []
    running_risk_set = sorted_risk_scores #not exp for DSL
    for t, count in zip(unique_times, counts):
        dN = sum(sorted_indicators[:count])
        denom = sum(running_risk_set)
        H0_t = dN / (denom+eps)
        H0.append(H0_t)
        
        # Update the running risk set
        running_risk_set = running_risk_set[count:]
        sorted_times = sorted_times[count:]
        sorted_indicators = sorted_indicators[count:]

    H0_cumulative = np.cumsum(H0)
    return unique_times, H0_cumulative


def sksurv_transform(E_data, Y_data):
    dt = np.dtype([('cens', '?'), ('time', '<f8')])
    y_new = np.array([tuple((bool(E),Y)) for E,Y in zip(E_data,Y_data)],dtype=dt)
    return y_new


def batched_predict(model, x_train, batch_size=None, predict_function=False):
    """
    Perform batch-based prediction using a model on input data.

    Args:
        model: The model for prediction.
        x_train: The input data for prediction.
        batch_size: The batch size for processing data in batches.
        predict_function: A custom predict function if needed.

    Returns:
        risk_scores: The predicted risk scores.
    """
    # Get the total number of samples
    num_samples = x_train.shape[0]
    if batch_size == None:
        batch_size = num_samples
    # Initialize an empty list to store the risk scores
    risk_scores = []

    # Iterate over the data in batches
    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_data = x_train[start_idx:end_idx]

        if predict_function:
            pred_output = model.predict(batch_data)
            if torch.is_tensor(pred_output):
                pred_output = pred_output.detach().cpu()
            batch_scores = np.array(pred_output) + 1e-5
        else:
            batch_scores = model(batch_data)
            if torch.is_tensor(batch_scores):
                batch_scores = batch_scores.detach().cpu()
            batch_scores = np.array(batch_scores) + 1e-5

        # Append the batch scores to the results
        risk_scores.extend(batch_scores)

    # Convert the list of batch results to a numpy array
    risk_scores = np.array(risk_scores)

    return risk_scores


def sksurv_metrics(model, x_train, y_train, x_test, y_test, predict_function=False, use_exp=True, eps=1e-3, n_band_split=20, plot=False, model_name = "Default"):
    model.trainable=False
            
    if len(y_train)==2:
        Y_train, E_train = y_train
        Y_test, E_test = y_test
    elif len(y_train)==3:
        Y_train, E_train, J_train = y_train
        Y_test, E_test, J_test = y_test
        
    y_train = sksurv_transform(E_train,Y_train)
    y_test = sksurv_transform(E_test,Y_test)
    
    if torch.is_tensor(Y_train):
        Y_train, E_train, Y_test, E_test = Y_train.cpu().numpy(), E_train.cpu().numpy(), Y_test.cpu().numpy(), E_test.cpu().numpy()
        
    if len(x_train.shape)>2:
        batch_size = len(x_train)//20
    else:
        batch_size = len(x_train)
    risk_scores = batched_predict(model, x_train, batch_size=batch_size, predict_function=predict_function)
    if use_exp:
        risk_scores=np.exp(risk_scores)
    # Calculate the Breslow estimator for unique times in the training set
    unique_times, H0_cumulative = breslow_estimator(Y_train, E_train, risk_scores)

    # Compute the relative hazard for the test set
    if len(x_train.shape)>2:
        batch_size = 4
    else:
        batch_size = len(x_test)
    relative_hazard = batched_predict(model, x_test, batch_size=batch_size, predict_function=predict_function) 
    if use_exp: 
        relative_hazard = np.exp(relative_hazard)

    start, end = Y_test.min(), Y_test.max()
    bandwidth = (end-start)/n_band_split  # Adjust based on your dataset
    time_grid = np.linspace(start+eps, end-eps, num=100)

    smoothed_H0 = [kernel_smoothed_hazard(t, unique_times, H0_cumulative, bandwidth) for t in time_grid]
    smoothed_H0 = np.array(smoothed_H0)

    # Calculate the cumulative hazard for the test set
    X_cumhazard = np.matmul(relative_hazard.reshape(-1,1),smoothed_H0.reshape(1, -1))

    # Calculate the survival matrix for the test set
    survival_matrix = np.exp(-X_cumhazard.astype(float)) #shape: n_test x n_times
    
    IBS = integrated_brier_score(y_train, y_test, survival_matrix, time_grid)
    
    # comp_surv(Y_train, E_train, Y_test, E_test, survival_matrix)
    AUC = cumulative_dynamic_auc(y_train, y_test, X_cumhazard, time_grid)
    if torch.is_tensor(E_test):
        E_test = E_test.detach().numpy()
    CI = concordance_index_censored(E_test.astype(bool), Y_test, relative_hazard.squeeze())
    if plot:
        metrics = f"CI={CI[0]}  IBS={IBS}  AUC={AUC[1]}"
        times, BS = brier_score(y_train, y_test, survival_matrix, time_grid)
        comp_surv(Y_train, E_train, Y_test, E_test, survival_matrix, times, BS, metrics, model_name=model_name)
    MSE = surv_rmse(Y_test, E_test, survival_matrix, time_grid)
        
    return AUC[1], IBS, CI[0] , MSE



def discrete_metrics(surv, y_train, y_test, eps=1e-5, n_band_split=1, num=100, plot=False, model_name="Default"):
    # Suppose you have input and output arrays from surv (df frame)
    input_array = np.array(surv.index)
    output_array = np.array((surv.values).T)
    
    Y_test, E_test = y_test[0], y_test[1]
    Y_train, E_train = y_train[0], y_train[1]
    # Define the new input values at which you want to interpolate
    start, end = Y_test.min(), Y_test.max()
    bandwidth = (end-start)/n_band_split  # Adjust based on your dataset
    time_grid = np.linspace(start+eps, end-eps, num=num)
    survival_matrix = np.zeros((len(output_array), num))

    # Use numpy.interp to perform linear interpolation
    for i in range(len(output_array)):
        survival_matrix[i,:] = np.interp(time_grid, input_array,  output_array[i,:])

    # Calculate the cumulative hazard for the test set
    X_cumhazard = -np.log(survival_matrix+eps)
    
    Y_train, E_train = y_train[0], y_train[1]
    
    y_test =  sksurv_transform(E_test, Y_test)
    y_train =  sksurv_transform(E_train, Y_train)
    IBS = integrated_brier_score(y_train, y_test, survival_matrix,time_grid)   
    ev = EvalSurv(surv, Y_test, E_test, censor_surv='km')
    CI = ev.concordance_td('antolini')
    AUC = cumulative_dynamic_auc(y_train, y_test, X_cumhazard,time_grid)
    if plot:
        times, BS = brier_score(y_train, y_test, survival_matrix, time_grid)
        metrics = f"CI={CI}  IBS={IBS}  AUC={AUC[1]}"
        comp_surv(Y_train, E_train, Y_test, E_test, survival_matrix, times, BS, metrics, model_name=model_name)
    MSE = surv_rmse(Y_test, E_test, survival_matrix, time_grid)
    return AUC[1],  IBS, CI , MSE


def DSL_metrics(model, weights_share, x_train, y_train, x_test, y_test, eps=1e-4, n_band_split=50, plot=False):
    model.trainable = False
    Y_train, E_train, C_train = y_train
    Y_test, E_test, C_test =y_test
    y_train, y_test = sksurv_transform(E_train, Y_train), sksurv_transform(E_test, Y_test)
    risk_scores = np.array(model(x_train)) + 1e-5
    relative_hazard = model(x_test)
    relative_hazard = np.array(relative_hazard)
    start, end = Y_test.min(), Y_test.max()
    bandwidth = (end - start) / n_band_split  # Adjust based on your dataset
    time_grid = np.linspace(start + eps, end - eps, num=100)
    smoothed_H0, H0_cumu, unique_times, smooth_hazards = smooth_subgroup_H0(subgroups, Y_train, E_train, C_train, risk_scores, bandwidth, time_grid)
    # smoothed_H0 = estimate_cumulative_hazard(Y_train, E_train, time_grid)
    # class imbalance for scores?
    # i = 0
    # for subgroup in subgroups:
    #     smoothed_H0[subgroup] = smoothed_H0[subgroup]*weights_share[i]
    #     i += 1
    X_cumhazard = np.zeros((len(E_test), len(time_grid)))
    # Calculate the cumulative hazard for the test set
    for i in range(len(E_test)):
        X_cumhazard[i] = relative_hazard[i] * smoothed_H0[C_test[i]]
    # Calculate the survival matrix for the test set
    survival_matrix = np.exp(-X_cumhazard)  # shape: n_test x n_times
    test_subindex = [np.argwhere(C_test == a) for a in subgroups]
    train_subindex = [np.argwhere(C_train == a) for a in subgroups]
    plt.clf()
    fig, axs = plt.subplots(2)
    fig.tight_layout()
    AUC = cumulative_dynamic_auc(y_train, y_test, X_cumhazard, time_grid)
    IBS = integrated_brier_score(y_train, y_test, survival_matrix, time_grid)
    CI = concordance_index_censored(E_test.astype(bool), Y_test, relative_hazard.squeeze())
    time_dep_relative_hazard = np.zeros((len(E_test), len(time_grid)))
    if plot:
        comp_surv(Y_train, E_train, Y_test, E_test, risk_scores, survival_matrix)
        plot_sub_surv(subgroups, Y_train, E_train, C_train, Y_test, E_test, C_test, risk_scores, survival_matrix, bandwidth, time_grid)

    return AUC[1], IBS, CI[0]
