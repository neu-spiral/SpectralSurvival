import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, ActivityRegularization
import numpy as np
from methods.utils import *
from methods.metrics import *
import scipy.sparse.linalg as spsl
import scipy.linalg as spl
from tensorflow.keras.optimizers import Nadam,Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import os
from methods.models import ResNet_3D, MLP, ResNet18_1D, Conv3D_torch
from tensorflow.keras.regularizers import l2

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import sys
import wandb


def torch_data(x_train, x_val, x_test, y_train, y_val, y_test, device=None):
    def to_tensor(x, device=None):
        x_np = np.array(x)
        return torch.tensor(x_np, dtype=torch.float32).to(device) if device else torch.tensor(x_np, dtype=torch.float32)
    # Convert NumPy arrays to PyTorch tensors
    x_train_tensor, x_val_tensor, x_test_tensor = map(lambda x: to_tensor(x, device), [x_train, x_val, x_test])
    y_train_tensor, y_val_tensor, y_test_tensor = map(lambda x: to_tensor(x, device), [y_train, y_val, y_test])
    return x_train_tensor, x_val_tensor, x_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor


class loss_kl_pi_pitilde(nn.Module):
    def __init__(self):
        super(loss_kl_pi_pitilde, self).__init__()

    def forward(self, inputs, targets):
        inputs, targets = inputs.squeeze(), targets.squeeze()
        global u_global
        n = len(targets)
        loss_a = torch.sum(u_global* inputs)/n
        loss_b = F.binary_cross_entropy(inputs, targets)
        return  loss_a + loss_b


def negative_log_likelihood(E):
    def loss(y_true, y_pred):
        '''y_true and y_pred should be of shape (n,)'''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, sort_idx = torch.sort(y_true, descending=True)
   
        y_true, y_pred = y_true[sort_idx], y_pred[sort_idx].reshape((-1,1))
        hazard_ratio = torch.exp(y_pred)
        
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0)).reshape((-1,1))
        uncensored_likelihood = torch.transpose(y_pred, 0, 1) - log_risk
        censored_likelihood = uncensored_likelihood * E[sort_idx]
        neg_likelihood_ = -torch.sum(censored_likelihood)
        neg_likelihood = neg_likelihood_ / len(E)
        return neg_likelihood.item()
    return loss

def save_model(model, path):
    state = {'model_state_dict': model.state_dict()}
    torch.save(state, path)
    
    
def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def evaluate(model, X, y, weights=None, eps=1e-5):
    Y, E = y
    if weights is None:
        with torch.no_grad():
            output = torch.squeeze(model(X)) + eps
    else:
        output = torch.squeeze(weights) + eps
    y_pred = torch.log(output)
    loss_function = negative_log_likelihood(E)
    loss = loss_function(Y, y_pred)
    torch.cuda.empty_cache()
    return loss


def statdist(generator, pepochs=200, method="power", v_init=None, ptol=0.00001):
    n = generator.shape[0]

    def power_iteration(A, v_init, epochs, tol):
        v = v_init 
        v = v / v.norm()
        for _ in range(epochs):
            Av = torch.matmul(A, v)
            v = Av / Av.norm()
            r = v - v_init if v_init is not None else Av
            normr = r.norm()
            if normr < tol:
                break
        return v

    if method == "kernel":
        lu, piv = torch.lu(generator.T)
        left = lu[:-1, :-1]
        right = -lu[:-1, -1]
        res = torch.solve(right.unsqueeze(-1), left)[0].squeeze()
        res = torch.cat((res, torch.tensor([1.0])))
        return (1.0 / res.sum()) * res

    if method == "eigenval":
        eps = 1.0 / torch.max(torch.abs(generator))
        mat = torch.eye(n) + eps * generator
        A = mat.t()
        eigvals, eigvecs = torch.symeig(A, eigenvectors=True)
        res = eigvecs[:, 0]
        return (1.0 / res.sum()) * res

    if method == "power":
        eps = 1.0 / torch.max(torch.abs(generator))
        mat = torch.eye(n).to(device) + eps * generator
        A = mat.t()
        v = v_init if v_init is not None else torch.rand(n)
        v = v / v.norm()
        v = power_iteration(A, v.to(device), pepochs, ptol)
        res = v.real
        return (1.0 / res.sum()) * res

    raise RuntimeError("not (yet?) implemented")
        
def ilsrx(y_train, rho, weights, x_beta_b, u, monitor_loss, ranking, classed=False, inner_params=[100, 1e-4], eps=1e-5):
    weights, x_beta_b, u = weights.squeeze(), x_beta_b.squeeze(), u.squeeze()
    iepochs = inner_params[0]
    rtol = inner_params[1]

    Y_train, E_train = y_train
    if classed:
        # Implement the get_reweights function for PyTorch if needed
        reweight_matrix_np = get_reweights(x_train.cpu().detach().numpy(), Y_train.cpu().detach().numpy(), E_train.cpu().detach().numpy(), C_train.cpu().detach().numpy(), subgroups, weights.cpu().detach().numpy())
        reweight_matrix = torch.tensor(reweight_matrix_np).to(device)
    n = len(weights)
    ilsr_conv = False
    iter = 0
    epsilon = 0.0001
    while not ilsr_conv:
        # u is reversed
        sigmas = rho * (1 + torch.log(torch.div(weights.to(device), x_beta_b + epsilon) + epsilon) - u.to(device))
        pi_sigmas = weights.to(device) * sigmas

        ind_minus = torch.where(sigmas < 0)[0]
        ind_plus = torch.where(sigmas >= 0)[0]
        scaled_sigmas_plus = 2 * sigmas[ind_plus] / (torch.sum(pi_sigmas[ind_minus]) - torch.sum(pi_sigmas[ind_plus]))

        chain = torch.zeros((n, n), dtype=torch.float32).to(device)
        for ind_minus_cur in ind_minus:
            chain[ind_plus, ind_minus_cur] = pi_sigmas[ind_minus_cur] * scaled_sigmas_plus

        for i, winner in enumerate(ranking):
            if classed:
                winner_weight = reweight_matrix[i, i]
            sum_weights = sum(weights[x] for x in ranking[i:]) + eps
            for loser in ranking[i + 1:]:
                if classed:
                    loser_weight = reweight_matrix[i:, i]
                    val = winner_weight / torch.dot(loser_weight, torch.tensor([weights[x] for x in ranking[i:]], dtype=torch.float32))
                else:
                    val = 1 / sum_weights
                chain[loser, winner] += val

        chain -= torch.diag(chain.sum(dim=1))
        weights_prev = weights.clone()
        weights = statdist(chain, v_init=weights)
        
        ilsrx_pred = torch.log(weights + 1e-5)
        loss_function = negative_log_likelihood(E_train)
        ilsrx_loss = loss_function(Y_train, ilsrx_pred)
        monitor_loss["ilsrx_loss"].append(ilsrx_loss)

        # Check convergence
        iter += 1
        ilsr_conv = torch.norm(weights_prev.to(device) - weights) < rtol * torch.norm(weights) or iter >= iepochs

    return weights, chain, monitor_loss


def deepspectral(model, optimizer, x_train, x_val, x_test, y_train, y_val, y_test, rho, monitor_loss, ranking, weights=None, x_beta_b=None, u=None, classed=False, gamma=1, epoch=0, inner_params=[100, 1e-4], avg_window=5, eps=1e-5):
    depochs = inner_params[0]
    rtol = inner_params[1]

    # pi update: no matter what the initial weights are, should come first.
    weights, chain, monitor_loss = ilsrx(y_train, rho, weights, x_beta_b, u, monitor_loss, ranking, classed=classed, inner_params=inner_params)
    
    MSE_score = torch.sum((x_beta_b - weights)**2)
    weights = weights.unsqueeze(1).to(device)
    # weights = weights.detach()
    params_conv = False
    iter = 0
    window_losses = [100000]
    # criterion = nn.MSELoss()
    # plt.plot(model(x_train).detach().cpu().numpy(), label='outputs')
    criterion = loss_kl_pi_pitilde()
    while not params_conv:
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, weights.detach())
        # loss = criterion(outputs, weights)
        # print(f'training loss: {loss}')
        loss.backward()
        optimizer.step()

        window_losses.append(loss.item())
        x_beta_b = model(x_train)  # predict new scores
        
        iter += 1  # log number of epochs
    
        avg_loss = np.mean(window_losses[:-1]) if iter <= avg_window else np.mean(window_losses[-avg_window - 1:-1])

        params_conv = (np.abs(avg_loss - window_losses[-1]) < 10 * rtol * avg_loss) or iter >= depochs  # check conv.
    # plt.plot(weights.detach().cpu().numpy(), label='weights')
    # plt.plot(y_train[0].detach().cpu().numpy(), label='label')
    # plt.plot(x_beta_b.detach().cpu().numpy(), linewidth=2.0, label='new_outputs')
    # plt.legend()
    # sys.exit("test done")
    # Dual update
    u += gamma * rho * (x_beta_b - weights.to(device))  # reversed
    global u_global
    u_global = u.clone().detach().to(device)

    ilsrx_loss = evaluate(model, x_train, y_train, weights)
    val_loss = evaluate(model, x_val, y_val)
    train_loss = evaluate(model, x_train, y_train)

    monitor_loss["ilsrx_loss"].append(ilsrx_loss)
    monitor_loss["val_loss"].append(val_loss)
    monitor_loss["train_loss"].append(train_loss)
    monitor_loss["MSE_score"].append(MSE_score)
    monitor_loss["NNtrain_loss"].append(loss)
    
    return model, weights, x_beta_b, u, monitor_loss, chain


def admm_kl(model, optimizer, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, epochs, batch_size, classed, inner_params, patience=10):
    ranking = torch.argsort(y_train[0].squeeze(), descending=False)
    n = len(x_train)
    rho = 1
    weights = torch.ones(n, dtype=torch.float)/ n
    x_beta_b = weights.clone().detach().to(device)
    u = torch.zeros((n,1)).to(device)
    eps = 1e-5
    gamma = 1
    chain_list, weight_list, log = [], [], []
    monitor_loss = {}
    monitor_loss["val_loss"], monitor_loss["train_loss"], monitor_loss["ilsrx_loss"], monitor_loss["MSE_score"], monitor_loss["NNtrain_loss"] = [], [], [], [], []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f'./models/model_checkpoint_{timestamp}.pth'

    val_loss = evaluate(model, x_val, y_val)
    monitor_loss["val_loss"].append(val_loss)
    current_loss = val_loss
    save_model(model, checkpoint_path)
    print(f'val loss:{val_loss}')
    
    train_loss = evaluate(model, x_train, y_train)
    monitor_loss["train_loss"].append(train_loss)
    current_loss = float('inf')
    no_improvement_count = 0
    print(f'trian loss:{train_loss}')
    
    for i in range(epochs):
        gamma /= (i + 1)
        model, weights, x_beta_b, u, monitor_loss, chain = deepspectral(model, optimizer, x_train, x_val, x_test, y_train, y_val, y_test, rho, monitor_loss, ranking, weights, x_beta_b, u, classed, gamma, i, inner_params)
        chain_list.append(chain)
        weight_list.append(weights)
        print(f'val loss:{monitor_loss["val_loss"][-1]}')
        print(f'tran loss:{monitor_loss["train_loss"][-1]}')
        
        # Log training and validation metrics
        wandb.log({
            "epoch": i,
            "train_loss": monitor_loss["train_loss"][-1],
            "val_loss": monitor_loss["val_loss"][-1],
            "ilsrx_loss": monitor_loss["ilsrx_loss"][-1],
            "MSE_score": monitor_loss["MSE_score"][-1]
        })

        if monitor_loss["val_loss"][-1] < current_loss:
            save_model(model, checkpoint_path)
            current_loss = monitor_loss["val_loss"][-1]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
    
        if no_improvement_count >= patience:
            print(f'Early stopping at epoch {i} as validation loss did not improve for {patience} consecutive epochs.')
            break
            
    model = load_model(model, checkpoint_path).cpu()
    x_train = x_train.detach().cpu()
    y_train = y_train.detach().cpu()
    x_val = x_val.detach().cpu()
    y_val = y_val.detach().cpu()
    x_test = x_test.detach().cpu()
    y_test = y_test.detach().cpu()
    if classed:
        # Replace DSL_metrics with the corresponding PyTorch implementation
        train_subindex = [np.argwhere(C_train == a) for a in subgroups]
        weights_share = []
        for i in range(len(subgroups)):
            weights_share.append(sum(weights[train_subindex[i]]))
        test_AUC, test_IBS, test_CI = DSL_metrics(model, weights_share, x_train, y_train, x_test, y_test)
        val_AUC, val_IBS, val_CI = DSL_metrics(model, weights_share, x_train, y_train, x_val, y_val)
    else:        
        test_AUC, test_IBS, test_CI = sksurv_metrics(model, x_train, y_train, x_test, y_test, use_exp=False)
        val_AUC, val_IBS, val_CI = sksurv_metrics(model, x_train, y_train, x_val, y_val, use_exp=False)
        print("test AIC:", test_AUC, test_IBS, test_CI, "val AIC:", val_AUC, val_IBS, val_CI)
    
    os.remove(checkpoint_path)
    log = [monitor_loss["NNtrain_loss"], monitor_loss["val_loss"], monitor_loss["train_loss"], monitor_loss["ilsrx_loss"], monitor_loss["MSE_score"], chain_list, weight_list]
    
    return test_AUC, test_IBS, test_CI, val_AUC, val_IBS, val_CI, log


def run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params=[100, 1e-4], l2_reg=2, classed=False):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_val, x_test, y_train, y_val, y_test = torch_data(x_train, x_val, x_test, y_train, y_val, y_test, device=device)
    activation = "relu"

    if len(x_train.shape) == 2:
        n_features = x_train.shape[1]
        if ResNet:
            x_train = x_train.unsqueeze(1)
            x_test = x_test.unsqueeze(1)
            x_val = x_val.unsqueeze(1)
            model = ResNet1D_tf(sigmoid=True)
        else:
            model = MLP1D_tf(dims[0], activation, n_features, depth, dropout)
    else:
        if ResNet:
            model = Conv3D_torch(out_features=1, depth=64)
        else:
            model = ResNet3D_tf(18)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=l2_reg)
    global u_global
    u_global = torch.zeros(len(x_train), dtype=torch.float).to(device)
    
    test_AUC, test_IBS, test_CI, val_AUC, val_IBS, val_CI, log = admm_kl(model, optimizer, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, epochs, batch_size, classed, inner_params)

    return test_AUC, test_IBS, test_CI, val_AUC, val_IBS, val_CI, log

def run_dsl(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params=[100, 1e-4], l2_reg=2, classed=False):
    return run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params, l2_reg, classed)

def run_cdsl(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params=[100, 1e-4], l2_reg=2):
    return run_model(model, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout, batch_size, inner_params, l2_reg, classed=True)