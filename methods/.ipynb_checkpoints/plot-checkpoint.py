from matplotlib import pyplot as plt
from methods.utils import *
from methods.metrics import *
import numpy as np
import os



def emp_surv(Y_train, E_train, time_grid, label = "train emp survival", ax=None, line_style='--', plot=False):
    # Initialize arrays to store survival probabilities and counts at risk
    survival_probabilities = []
    counts_at_risk = []
    for t in time_grid:
        n_happened = 0
        count_at_risk = len(Y_train)
        # Calculate the number of events at time t
        for i in range(len(Y_train)):
            if Y_train[i] <= t and E_train[i] == 1:
                n_happened += 1
    
        # Update the count at risk
        # print(count_at_risk)
        # Calculate the survival probability at time t
        survival_probability = (len(Y_train)-n_happened) / len(Y_train)

        # Append the values to the respective arrays
        survival_probabilities.append(survival_probability)
        counts_at_risk.append(count_at_risk)

    # Plot the empirical survival function
    if plot:
        if ax:
            ax.plot(time_grid, survival_probabilities,linestyle=line_style, label = label)
            ax.grid(True)
        else:
            plt.plot(time_grid, survival_probabilities,linestyle=line_style, label = label)
            plt.title('Empirical Survival Function')
            plt.xlabel('Time')
            plt.ylabel('Survival Probability')
            plt.grid(True)
    return survival_probabilities



# plt.show()
def comp_surv(Y_train, E_train, Y_test, E_test, survival_matrix, times, BS, metrics, model_name="Default", eps=1e-5):
    start, end = 0, Y_test.max()
    time_grid = np.linspace(start + eps, end - eps, num=100)
    
    plt.figure(figsize=(10, 6))
    
    train_sur = emp_surv(Y_train, E_train, time_grid, plot=True)
    test_sur = emp_surv(Y_test, E_test, time_grid, "test emp survival", line_style="-.", plot=True)
    MSE = np.mean((np.mean(survival_matrix, axis=0)-test_sur)**2)
    plt.plot(time_grid, np.mean(survival_matrix, axis=0), label="est overall survival")
    plt.plot(times, BS, label="Brier score", linestyle=':')
    plt.legend()
    # Add epoch information to the title
    plt.title(f'Comparison of Survival Curves ({model_name}) MSE({MSE})'+metrics)
    
    counter = 0
    filename = "./experiment_results/comp_surv{}_{}.png"
    while os.path.isfile(filename.format(counter, model_name)):
        counter += 1
    filename = filename.format(counter, model_name)
    plt.savefig(f'{filename}')

    
def plot_sub_surv(subgroups, Y_train, E_train, C_train, Y_test, E_test, C_test, risk_scores, survival_matrix, bandwidth, time_grid, eps=1e-5):
    chars = [
        {'label': 'est-GCB', 'color': 'blue', 'linestyle': '-', 'marker': 'o'},
        {'label': 'est-ABC', 'color': 'red', 'linestyle': '--', 'marker': 's'},
        {'label': 'est-TP3', 'color': 'green', 'linestyle': '-.', 'marker': '^'},
    ]


   
    num_subplots = len(chars)
    start, end = 0, Y_test.max()
    time_grid = np.linspace(start + eps, end - eps, num=100)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6*num_subplots), sharex=True)
    test_subindex = [np.argwhere(C_test == a) for a in subgroups]
    train_subindex = [np.argwhere(C_train == a) for a in subgroups]
    i=0
    for subgroup in subgroups:
        char = chars[i]
        ax = axes[i]
        ax.plot(time_grid, np.mean(survival_matrix[test_subindex[i]],axis=0).squeeze(), label=char['label'])
        emp_surv(Y_test[test_subindex[i]], E_test[test_subindex[i]],time_grid, label="test emp survival", ax=ax, line_style = "-.")
        emp_surv(Y_train[train_subindex[i]], E_train[train_subindex[i]], time_grid,ax=ax)
        ax.set_xlabel('time')
        ax.set_ylabel('Survival rate')
        ax.legend()
        i += 1
    counter = 0
    filename = "./experiment_results/sub_surv{}.png"
    while os.path.isfile(filename.format(counter)):
        counter += 1
    filename = filename.format(counter)
    plt.savefig(f'{filename}')