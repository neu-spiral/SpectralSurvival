import numpy as np
import os
from SurvSet.data import SurvLoader
import math
from lifelines import utils
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import train_test_split
import torch
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, concordance_index_censored
from deepsurvk.datasets import load_whas, load_rgbsg, load_simulated_gaussian, load_metabric, load_simulated_linear, load_simulated_treatment, load_support
import seaborn as sns
import inspect
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.model_selection import KFold
from methods.preprocessing import prep_data, preprocessing, load_lung1, load_syn_ads, prepare_ads, get_journey_rankings, load_BraTS
import subprocess
import wandb


def kernel_smoothed_hazard(t, observed_times, observed_hazards, bandwidth, eps=1e-5):
    # Calculate the kernel weights using the Epanechnikov kernel
    weights = epanechnikov_kernel(t, observed_times, bandwidth)
    if isinstance(weights, torch.Tensor):
        weights = weights.numpy()
    # print(f"w:{weights}")
    # Check if the sum of weights is close to zero (nearby times have very small weights)
    if sum(weights) <= eps:
        # If so, return the last observed hazard as a fallback
        if np.sum(observed_times <= t)==0:
            return 0
        else:
            last_index = np.max(np.where(observed_times <= t))
            return observed_hazards[last_index]
    else:
        # Normalize the weights to sum to 1
        weights /= sum(weights)
        
        # Calculate the smoothed hazard by taking the weighted sum of observed hazards
        smoothed_hazard = np.dot(weights, observed_hazards)
        
        return smoothed_hazard


def epanechnikov_kernel(t, T, bandwidth):
    M = 0.75 * (1 - ((t - T) / bandwidth) ** 2)
    if torch.is_tensor(bandwidth):
        M[torch.tensor(abs((t - T))) >= bandwidth] = 0
    else:
        M[abs((t - T)) >= bandwidth] = 0
    return M


def sksurv_transform(E_data, Y_data):
    dt = np.dtype([('cens', '?'), ('time', '<f8')])
    y_new = np.array([tuple((bool(E),Y)) for E,Y in zip(E_data,Y_data)],dtype=dt)
    return y_new

def p(*args):
    # print variables with names
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    vnames = r.split(", ")
    for i,(var,val) in enumerate(zip(vnames, args)):
        print(f"{var} = {val}")
        
        
# unique is reversed VS Y_train
def reducing_tie(Y_train, limit):
    # eliminate time spots that have too many events
    unique, counts = np.unique(Y_train, return_counts=True)
    indeces=[]
    for i in range(len(unique)):
        if counts[i]<=limit:
            indeces.extend([1]*counts[i])
        else:
            indeces.extend([1]*limit)
            indeces.extend([0]*(counts[i]-limit))
    return np.array(indeces)[::-1]


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
        risk_set_idx = sorted_times >= t
        dN = sum(sorted_indicators[risk_set_idx][:count])
        denom = sum(running_risk_set[risk_set_idx])
        # if denom==0:
        #     print(t, risk_set_idx, denom)
        H0_t = dN / (denom+eps)
        H0.append(H0_t)
        
        # Update the running risk set
        running_risk_set = running_risk_set[risk_set_idx][count:]
        sorted_times = sorted_times[risk_set_idx][count:]
        sorted_indicators = sorted_indicators[risk_set_idx][count:]
    
    H0_cumulative = np.cumsum(H0)
    return unique_times, H0_cumulative

def get_count_list(Y_train):
    unique, counts = np.unique(Y_train, return_counts=True)
    count_list = []
    for count in counts:
        if count==1:
            count_list.append(1)
        else:
            for i in range(count):
                count_list.append(count-i)
            # count_list.append(count)
    # append 0 for the last one
    count_list.append(0)
    return count_list   


def estimate_cumulative_hazard(Y_train, E_train, time_grid):
    """
    Estimate the baseline cumulative hazard rate at specified time points using Kaplan-Meier estimator.

    Parameters:
        - data: DataFrame with 'time' and 'event' columns.
        - time_grid: List or array of time points at which to estimate the cumulative hazard rate.

    Returns:
        - cumulative_hazard: NumPy array containing estimated cumulative hazard rate at each time point.
    """
    # Initialize the Kaplan-Meier estimator
    kmf = KaplanMeierFitter()
    # Fit the estimator to the data
    kmf.fit(Y_train, E_train)
    # Estimate cumulative hazard rate at specified time points
    cumulative_hazard = -np.log(kmf.survival_function_at_times(time_grid).values)
    return cumulative_hazard


def smooth_subgroup_H0(subgroups, Y_train, E_train, C_train, risk_scores, bandwidth, time_grid):
    train_subindex = [np.argwhere(C_train == a) for a in subgroups]
    unique_times, H0_cumu = {}, {}
    count = 0
    smooth_H0 = {}
    delta_H0 = {}
    smooth_hazards = {}
    for subgroup in subgroups:
        unique_times[subgroup], H0_cumu[subgroup] = \
            breslow_estimator(Y_train[train_subindex[count].squeeze()], E_train[train_subindex[count].squeeze()],
                              risk_scores[train_subindex[count].squeeze()])
        unique_times[subgroup] = np.insert(unique_times[subgroup], 0, 0)
        H0_cumu[subgroup] = np.insert(H0_cumu[subgroup], 0, 0)
        delta_H0[subgroup] = np.diff(np.concatenate(([0], H0_cumu[subgroup])))
        smooth_hazards[subgroup] = np.array(
            [kernel_smoothed_hazard(t, unique_times[subgroup], delta_H0[subgroup], bandwidth) for t in time_grid])
        smooth_H0[subgroup] = np.array(
            [kernel_smoothed_hazard(t, unique_times[subgroup], H0_cumu[subgroup], bandwidth) for t in time_grid])
        smooth_H0[subgroup][0] = 0
        count += 1
    return smooth_H0, H0_cumu, unique_times, smooth_hazards

def compute_reweight_matrix(X_train, C_train, smooth_hazards, eps=1e-5):
    n = len(X_train)
    weight_matrix = np.zeros((n, n))
    for i in range(n):
        weight_matrix[i, :] = np.array(smooth_hazards[C_train[i]])+eps
    reweight_matrix = np.flip(weight_matrix, 1)
    return reweight_matrix


def get_reweights(X_train, Y_train, E_train, C_train, subgroups, risk_scores, n_band_split=50):
    start, end = Y_train.min(), Y_train.max()
    bandwidth = (end - start) / n_band_split
    smooth_H0, H0_cumu, unique_times, smooth_hazards = smooth_subgroup_H0(subgroups, Y_train, E_train, C_train, risk_scores, bandwidth, Y_train)
    reweight_matrix = compute_reweight_matrix(X_train, C_train, smooth_hazards, eps=1e-5)
    return reweight_matrix


def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df


def cross_validate(ResNet, algo, dataset, learning_rate, dims, depth, epochs, dropout, \
                   inner_params=[100, 1e-4], n_folds=5, l2_reg=2, batch_size_coef=0.1, n=1000, tensor=False):    
    config = {
        "ResNet": ResNet,
        "dataset": dataset,
        "learning_rate": learning_rate,
        "dims": dims,
        "depth": depth,
        "epochs": epochs,
        "dropout": dropout,
        "batch_size_coef": batch_size_coef,
        "n_folds": n_folds,
        "n": n,
        "tensor": tensor,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_metrics = []
    if dataset == "syn_ads":
        m, s, feature_dim = 50, 200, 50  # adjust these values as needed
        file_name = f'./datasets/Synthetic/synthetic_n{n}_m{m}_s{s}_dim{feature_dim}.pickle'
        df = pd.read_pickle(file_name) 
        # The train val test are splited by ads and we can not use CV here
        fold_train, fold_val, df_test = df["train"], df["val"], df["test"] 
        data_train = fold_train   
    elif dataset == "lung1":
        x_train, x_test, y_train, y_test = load_lung1(tensor)
        data_train = x_train
    elif dataset == "BraTS":
        x_train, x_test, y_train, y_test = load_BraTS(tensor)
        data_train = x_train
    else:
        df_train, df_test = prep_data(dataset)
        data_train = df_train
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_train)):
        wandb.init(project=f"Survival_analysis_{dataset}_cv", name=f"test_Spectral_Res{ResNet}_{depth}_{learning_rate}_{batch_size_coef}_fold{fold}", config=config)
        if dataset == "syn_ads":
            
            rankings = get_journey_rankings(fold_train)
            x_train_f, x_train, x_val, x_test, y_train_f, y_train, y_val, y_test = prepare_ads(fold_train, fold_val, df_test, tensor=True)
        
            batch_size = round(len(x_train) * batch_size_coef)
            
            test_AUC, test_IBS, test_CI, test_MSE, val_AUC, val_IBS, val_CI, val_MSE, log = \
                algo(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, rankings, inner_params, full_dataset=(x_train_f, y_train_f))
            
        else:
            if dataset == "lung1":
                x_train, x_test, y_train, y_test = load_lung1(tensor)
                x_train, x_val, x_test, y_train, y_val, y_test = x_train[train_idx], x_train[val_idx], x_test, (y_train[0][train_idx], y_train[1][train_idx]), (y_train[0][val_idx], y_train[1][val_idx]), y_test
                    
            elif dataset == "BraTS":
                x_train, x_test, y_train, y_test = load_BraTS(tensor)
                x_train, x_val, x_test, y_train, y_val, y_test = x_train[train_idx], x_train[val_idx], x_test, (y_train[0][train_idx], y_train[1][train_idx]), (y_train[0][val_idx], y_train[1][val_idx]), y_test

            else:
                fold_train = df_train.iloc[train_idx]
                fold_val = df_train.iloc[val_idx]
                x_train, x_val, x_test, y_train, y_val, y_test = preprocessing(dataset, fold_train, fold_val, df_test)
            
            batch_size = round(len(x_train) * batch_size_coef)
            rankings = [np.argsort(y_train[0].squeeze())[::1]]
        
            x_train, x_val, x_test = map(lambda x: torch.from_numpy(np.array(x)), [x_train, x_val, x_test])  #if isinstance(x, list) else x
            y_train = (torch.from_numpy(np.array(y_train[0])), torch.from_numpy(np.array(y_train[1])))
            y_val = (torch.from_numpy(np.array(y_val[0])), torch.from_numpy(np.array(y_val[1])))
            y_test = (torch.from_numpy(np.array(y_test[0])), torch.from_numpy(np.array(y_test[1])))

            # Move data to device
            x_train, x_val, x_test = x_train.to(device), x_val.to(device), x_test.to(device)
            y_train = (y_train[0].to(device), y_train[1].to(device))
            y_val = (y_val[0].to(device), y_val[1].to(device))
            y_test = (y_test[0].to(device), y_test[1].to(device))
            
            test_AUC, test_IBS, test_CI, test_MSE, val_AUC, val_IBS, val_CI, val_MSE, log = \
                algo(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, rankings, inner_params)
            
            
        wandb.log({
            f"fold_{fold+1}_test_AUC": test_AUC, f"fold_{fold+1}_test_IBS": test_IBS,
            f"fold_{fold+1}_test_CI": test_CI, f"fold_{fold+1}_test_MSE": test_MSE,
            f"fold_{fold+1}_val_AUC": val_AUC, f"fold_{fold+1}_val_IBS": val_IBS,
            f"fold_{fold+1}_val_CI": val_CI, f"fold_{fold+1}_val_MSE": val_MSE,
        })
        wandb.finish()
        # Save metrics for the fold
        fold_metrics.append((test_AUC, test_IBS, test_CI, test_MSE, val_AUC, val_IBS, val_CI, val_MSE))
        
    # Calculate mean and std across folds
    print(fold_metrics)
    if len(fold_metrics) == 1:
        mean_metrics = fold_metrics[0]
        std_metrics = [0] * 8
    else:
        mean_metrics = np.mean(fold_metrics, axis=0)
        std_metrics = np.std(fold_metrics, axis=0)
    
    wandb.init(project=f"DSL_{dataset}_mean", name=f"Spectral_{depth}_{learning_rate}_{batch_size_coef}_{fold}", config=config)
    wandb.log({
        "mean_test_AUC": mean_metrics[0], "mean_test_IBS": mean_metrics[1],
        "mean_test_CI": mean_metrics[2], "mean_test_MSE": mean_metrics[3],
        "mean_val_AUC": mean_metrics[4], "mean_val_IBS": mean_metrics[5],
        "mean_val_CI": mean_metrics[6], "mean_val_MSE": mean_metrics[7],
        "std_test_AUC": std_metrics[0], "std_test_IBS": std_metrics[1],
        "std_test_CI": std_metrics[2], "std_test_MSE": std_metrics[3],
        "std_val_AUC": std_metrics[4], "std_val_IBS": std_metrics[5],
        "std_val_CI": std_metrics[6], "std_val_MSE": std_metrics[7],
    })
    wandb.finish()
    return mean_metrics, std_metrics, log


def get_gpu_memory_usage(gpu_id):
    command = ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader', f'--id={gpu_id}']
    result = subprocess.run(command, stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip().split('\n')[0]
    used_memory, total_memory = map(int, output.split(','))
    return used_memory, total_memory


def cross_validate_baseline(ResNet, algo, dataset, learning_rate, dims, depth, epochs, dropout, batch_size_coef=0.1,
                   n_folds=5, n=1000, tensor=True):
    config = {
        "ResNet": ResNet,
        "dataset": dataset,
        "learning_rate": learning_rate,
        "dims": dims,
        "depth": depth,
        "epochs": epochs,
        "dropout": dropout,
        "batch_size_coef": batch_size_coef,
        "n_folds": n_folds,
        "n": n,
        "tensor": tensor,
    }
    
    if dataset == "lung1":
        x_train, x_test, y_train, y_test = load_lung1(tensor)
        data_train = x_train
    elif dataset == "BraTS":
        x_train, x_test, y_train, y_test = load_BraTS(tensor)
        data_train = x_train
    elif dataset == "syn_ads":
        m, s, feature_dim = 50, 200, 50
        file_name = f'./datasets/Synthetic/synthetic_n{n}_m{m}_s{s}_dim{feature_dim}.pkl'
        df =  pd.read_pickle(file_name)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        data_train = df_train
    else:
        df_train, df_test = prep_data(dataset)
        data_train = df_train
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_train)):
        wandb.init(project=f"Survival_analysis_{dataset}_cv", name=f"{algo.__name__}_Res{ResNet}_{depth}_{learning_rate}_{batch_size_coef}_{fold}", config=config)
        if dataset == "lung1":
            x_train, x_test, y_train, y_test = load_lung1(tensor)
            x_train, x_val, x_test, y_train, y_val, y_test = x_train[train_idx], x_train[val_idx], x_test, (
            y_train[0][train_idx], y_train[1][train_idx]), (y_train[0][val_idx], y_train[1][val_idx]), y_test
        elif dataset == "BraTS":
            x_train, x_test, y_train, y_test = load_BraTS(tensor)
            x_train, x_val, x_test, y_train, y_val, y_test = x_train[train_idx], x_train[val_idx], x_test, (
            y_train[0][train_idx], y_train[1][train_idx]), (y_train[0][val_idx], y_train[1][val_idx]), y_test
        elif dataset == "syn_ads":
            fold_train = df_train.iloc[train_idx]
            fold_val = df_train.iloc[val_idx]
            
        else:
            fold_train = df_train.iloc[train_idx]
            fold_val = df_train.iloc[val_idx]
            x_train, x_val, x_test, y_train, y_val, y_test = preprocessing(dataset, fold_train, fold_val, df_test)
            if ResNet:
                x_train, x_val, x_test = np.expand_dims(x_train,1), np.expand_dims(x_val,1), np.expand_dims(x_test,1)
        batch_size = round(len(x_train) * batch_size_coef)
        test_AUC, test_IBS, test_CI, test_MSE, val_AUC, val_IBS, val_CI, val_MSE, log = \
            algo(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, \
                 learning_rate, dims, depth, epochs, dropout, batch_size)
        
        fold_metrics.append((test_AUC, test_IBS, test_CI, test_MSE, val_AUC, val_IBS, val_CI, val_MSE))

        # Log metrics for the fold in wandb
        wandb.log({
            f"fold_{fold+1}_test_AUC": test_AUC, f"fold_{fold+1}_test_IBS": test_IBS,
            f"fold_{fold+1}_test_CI": test_CI, f"fold_{fold+1}_test_MSE": test_MSE,
            f"fold_{fold+1}_val_AUC": val_AUC, f"fold_{fold+1}_val_IBS": val_IBS,
            f"fold_{fold+1}_val_CI": val_CI, f"fold_{fold+1}_val_MSE": val_MSE,
        })

        # Save metrics for the fold
        fold_metrics.append((test_AUC, test_IBS, test_CI, test_MSE, val_AUC, val_IBS, val_CI, val_MSE))
        wandb.finish()
        
    # Calculate mean and std across folds
    if len(fold_metrics) == 1:
        mean_metrics = np.array(fold_metrics)
        std_metrics = [0] * 8
    else:
        mean_metrics = np.mean(fold_metrics, axis=0)
        std_metrics = np.std(fold_metrics, axis=0)
    # Log final metrics to wandb
    wandb.init(project=f"Survival_analysis_{dataset}_mean", name=f"{algo}_{depth}_{learning_rate}_{batch_size_coef}_{fold}", config=config)
    
    wandb.log({
        "mean_test_AUC": mean_metrics[0], "mean_test_IBS": mean_metrics[1],
        "mean_test_CI": mean_metrics[2], "mean_test_MSE": mean_metrics[3],
        "mean_val_AUC": mean_metrics[4], "mean_val_IBS": mean_metrics[5],
        "mean_val_CI": mean_metrics[6], "mean_val_MSE": mean_metrics[7],
        "std_test_AUC": std_metrics[0], "std_test_IBS": std_metrics[1],
        "std_test_CI": std_metrics[2], "std_test_MSE": std_metrics[3],
        "std_val_AUC": std_metrics[4], "std_val_IBS": std_metrics[5],
        "std_val_CI": std_metrics[6], "std_val_MSE": std_metrics[7],
    })
    wandb.finish()
    

    return mean_metrics, std_metrics, log