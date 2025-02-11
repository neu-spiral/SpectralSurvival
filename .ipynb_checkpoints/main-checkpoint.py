import argparse
import numpy as np
import pandas as pd
import json
import time
from deepsurvk.datasets import load_whas, load_rgbsg,load_simulated_gaussian,load_metabric,load_simulated_linear,load_simulated_treatment,load_support
from datasets import metabric, support, whas
from methods.utils import *
from methods.baseline import *
from methods.survmodel_pytorch import *   
import wandb


np.random.seed(1234)
_ = torch.manual_seed(1234)  
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', nargs='?',action='store', type=str, help='dataset name: [DLBCL, vdv, DBCD, metabric, support, whas, lung1, syn_ads, BraTS]', default="DLBCL")
parser.add_argument("--algorithm", nargs='?', action="store", type=str, help='spectral or baseline', default="spectral")
parser.add_argument('--ResNet', action='store_true', help='Use ResNet18_1D with sigmoid activation')
parser.add_argument('--learning_rate',  nargs='?',help='learning rate', type=float, default=0.01)
parser.add_argument('--depth',  nargs='?',help='depth', type=int, default=3)
parser.add_argument('--dropout',  nargs='?',help='dropout', type=float, default=0.5)
parser.add_argument('--batch_size_coef',  nargs='?',help='batch_size_coef', type=float, default=1.0)
parser.add_argument('--rtol',  nargs='?',help='rtol', type=float, default=1e-4)
parser.add_argument('--n_iter',  nargs='?',help='rtol', type=float, default=100)
parser.add_argument('--epochs', nargs='?', type=int, default=200, help='Number of epochs')
parser.add_argument('--gpu_id',  nargs='?',help='gpu_id', type=int, default=0)


# Assign arguments to variables
args = parser.parse_args()
gpu_id = args.gpu_id
dataset = args.dataset
algorithm = args.algorithm
ResNet = args.ResNet
learning_rate = args.learning_rate
depth = args.depth
dropout = args.dropout
batch_size_coef = args.batch_size_coef
epochs = args.epochs
rtol = args.rtol
n_iter = args.n_iter
dims = (200,) * (depth - 1)
activation = 'relu'
l2_reg = 16


config = {
    "dataset": dataset,
    "algorithm": algorithm,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "dims": dims,
    "depth": depth,
    "dropout": dropout,
    "batch_size_coef": batch_size_coef,
    "ResNet": ResNet,
}


# Decide whether to run DSL or the baseline algorithms
if algorithm and algorithm.lower() == "spectral":
    print("Running Spectral algorithm...")
    algo_func = run_dsl
    algorithm_names = ["Spectral"]
    algorithms = [algo_func]
    use_baseline = False
elif algorithm == "baseline":
    print("Running baseline algorithms based on dataset...")
    use_baseline = True
    if dataset in ["lung1", "BraTS"]:
        algorithm_names = ["DeepSurv", "DeepHit", "MTLR"]
        algorithms = [run_deepsurv, run_deephit, run_MTLR]
    else:
        if ResNet:
            algorithm_names = ["DeepHit", "MTLR", "CoxCC", "DeepSurv"]
            algorithms = [run_deephit, run_MTLR, run_coxcc, run_deepsurv]
        else:
            algorithm_names = ["DeepHit", "MTLR", "CoxCC", "FastCPH", "DeepSurv", "CoxTime"]
            algorithms = [run_deephit, run_MTLR, run_coxcc, run_fastcph, run_deepsurv, run_coxtime]
elif algorithm:
    print(f"Running {algorithm}")
    use_baseline = True
    if algorithm.lower() == "deepsurv":
        algo_func = run_deepsurv
        algorithm_names = ["DeepSurv"]
    elif algorithm.lower() == "deephit":
        algo_func = run_deephit
        algorithm_names = ["DeepHit"]
    elif algorithm.lower() == "mtlr":
        algo_func = run_MTLR
        algorithm_names = ["MTLR"]
    elif algorithm.lower() == "coxcc":
        algo_func = run_coxcc
        algorithm_names = ["CoxCC"]
    elif algorithm.lower() == "fastcph":
        algo_func = run_fastcph
        algorithm_names = ["FastCPH"]
    elif algorithm.lower() == "coxtime":
        algo_func = run_coxtime
        algorithm_names = ["CoxTime"]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    algorithms = [algo_func]
    
    


# Iterate over algorithms and run them
for algo_func, algo_name in zip(algorithms, algorithm_names):
    start_time = time.time()
    params_record = []
    if use_baseline:
        # Use cross_validate_baseline for baseline algorithms
        mean_metrics, std_metrics, log = cross_validate_baseline(
            ResNet=ResNet,
            algo=algo_func,
            dataset=dataset,
            learning_rate=learning_rate,
            dims=dims,
            depth=depth,
            epochs=epochs,
            dropout=dropout,
            batch_size_coef=batch_size_coef,
            n_folds=5,
        )
    else:
        # Use cross_validate for DSL
        mean_metrics, std_metrics, log = cross_validate(
            ResNet=ResNet,
            algo=algo_func,
            dataset=dataset,
            learning_rate=learning_rate,
            dims=dims,
            depth=depth,
            epochs=epochs,
            dropout=dropout,
            l2_reg=16, 
            n_folds=5,
            batch_size_coef=batch_size_coef,
            tensor=True,
        )
    
    elapsed_time = time.time() - start_time
    used_memory, total_memory = get_gpu_memory_usage(gpu_id)
    print(f"{algo_name}: {mean_metrics}±{std_metrics}")
    print(f"time: {elapsed_time}")
    # Save results for this algorithm
    params_record.append([dataset, algo_name, mean_metrics, std_metrics, elapsed_time, used_memory])


    params_dict_list = []
    for entry in params_record:
        dataset, algorithm, mean_metrics, std_metrics, elapsed_time, used_memory = entry
        params_dict_list.append({
            'Dataset': dataset,
            'Algorithm': algorithm,
            'AUC': f"{mean_metrics[0]:.3f}\u00B1{std_metrics[0]:.3f}",
            'IBS': f"{mean_metrics[1]:.3f}\u00B1{std_metrics[1]:.3f}",
            'CI': f"{mean_metrics[2]:.3f}\u00B1{std_metrics[2]:.3f}",
            'surv_MSE': f"{mean_metrics[3]:.3f}\u00B1{std_metrics[3]:.3f}",
            'AUC_val': f"{mean_metrics[4]:.3f}\u00B1{std_metrics[4]:.3f}",
            'IBS_val': f"{mean_metrics[5]:.3f}\u00B1{std_metrics[5]:.3f}",
            'CI_val': f"{mean_metrics[6]:.3f}\u00B1{std_metrics[6]:.3f}",
            'surv_MSE_val': f"{mean_metrics[7]:.3f}\u00B1{std_metrics[7]:.3f}",
            'LR': f"{learning_rate}",
            'BS': f"{batch_size_coef}",
            'depth': f"{depth}",
            'dropout': f"{dropout}",
            'time': f"{elapsed_time}",
            'GPU': f"{used_memory}MiB"
        })

    df = pd.DataFrame(params_dict_list, index=[0])
    df_string = df.to_string(index=False)

    print(config)

    if use_baseline:
        file_name = f'./experiment_results/Baseline_ResNet={ResNet}_{dataset}.txt'
    else:
        file_name = f'./experiment_results/Spectral_ResNet={ResNet}_{dataset}.txt'

    with open(file_name, 'a') as file:
        file.write(df_string+ '\n\n')
    

