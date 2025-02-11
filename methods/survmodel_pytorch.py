import numpy as np
import pandas as pd
from methods.utils import *
from methods.metrics import *
import scipy.sparse.linalg as spsl
import scipy.linalg as spl
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
from methods import models
import sys
import shutil
import time
import wandb
from methods.medicalnet import generate_model


# Corrected vectorized implementation
def MC_from_ranking(chain, rankings, weights, device, eps=1e-5):
    for ranking in rankings:
        if isinstance(ranking, torch.Tensor):
            ranking_tensor = ranking.clone().detach().to(device)
        else:
            ranking_tensor = torch.tensor(ranking, device=device)
            
        r_len = len(ranking_tensor)

        # Compute cumulative sum of weights from each position to the end
        sum_weights = torch.cumsum(weights[ranking_tensor].flip(0), dim=0).flip(0) + eps

        # Use broadcasting to compute winner-loser relationships
        winner_matrix = ranking_tensor.view(-1, 1).expand(r_len, r_len)
        loser_matrix = ranking_tensor.view(1, -1).expand(r_len, r_len)

        # Mask for upper triangular (winner before loser)
        mask = torch.triu(torch.ones((r_len, r_len), device=device), diagonal=1).bool()

        # Compute the values to be added
        val_matrix = torch.zeros((r_len, r_len), device=device)
        val_matrix[mask] = 1.0 / sum_weights[:-1].repeat_interleave(torch.arange(r_len - 1, 0, -1, device=device))

        # Correct update to the chain matrix
        chain[loser_matrix[mask], winner_matrix[mask]] += val_matrix[mask]

    return chain

def get_intrinsic_hazard(model, x_train, batch_size=64):
    """
    Computes x_beta_b in batches to handle large datasets.

    Args:
        model: The PyTorch model to use for prediction.
        x_train: The input data (torch.Tensor).
        batch_size: The size of each batch.

    Returns:
        torch.Tensor: Concatenated predictions for the entire dataset.
    """
    x_beta_b = []
    num_samples = x_train.size(0)
    
    if len(x_train.shape) <= 3:
        x_beta_b = model(x_train)
    
    else:# Iterate through the data in batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch = x_train[start_idx:end_idx]

            with torch.no_grad():
                # Perform model prediction and squeeze the output
                batch_pred = model(batch)
    #             print(batch_pred.shape)
                x_beta_b.append(batch_pred.squeeze())

        # Concatenate all batch predictions
        x_beta_b = torch.cat(x_beta_b, dim=0)
    return x_beta_b


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, weights, u):
        self.x_data = x_data
        self.weights = weights
        self.u = u

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return (self.x_data[idx],
                self.weights[idx],
                self.u[idx])
    

def custom_loss_kl_pi_pitilde(weights, y_pred, u):
    bce_loss = nn.functional.binary_cross_entropy(y_pred, weights, reduction='mean')
    return torch.sum(u * y_pred) + bce_loss


def negative_log_likelihood(E):
    def loss(y_true, y_pred):
        # Sort y_true and y_pred based on y_true values in descending order
        sort_idx = torch.argsort(y_true, descending=True)
        y_true = y_true[sort_idx]
        y_pred = y_pred[sort_idx]
        E_sorted = E[sort_idx]

        hazard_ratio = torch.exp(y_pred)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = y_pred - log_risk
        censored_likelihood = uncensored_likelihood * E_sorted
        neg_likelihood_ = -torch.sum(censored_likelihood)
        neg_likelihood = neg_likelihood_ / len(E)
        return neg_likelihood

    return loss

def JNPLL(hazards, times, events, journey_ids):
    def NPLL(E):
        def loss(y_true, y_pred):
            y_true = torch.tensor(y_true, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            E_torch = torch.tensor(E, dtype=torch.float32)

            sort_idx = torch.argsort(y_true, descending=True)
            y_true = y_true[sort_idx]
            y_pred = y_pred[sort_idx]
            E_sorted = E_torch[sort_idx]

            hazard_ratio = torch.exp(y_pred)
            log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
            uncensored_likelihood = y_pred - log_risk
            censored_likelihood = uncensored_likelihood * E_sorted
            neg_likelihood_ = -torch.sum(censored_likelihood)
            neg_likelihood = neg_likelihood_ / len(E)
            return neg_likelihood

        return loss

    unique_journeys = torch.unique(journey_ids)
    total_loss = 0.0

    for journey in unique_journeys:
        mask = journey_ids == journey
        journey_hazards = hazards[mask]
        journey_times = times[mask]
        journey_events = events[mask]

        # Sort journey-specific data based on times in descending order
        sort_idx = torch.argsort(journey_times, descending=True)
        sorted_hazards = journey_hazards[sort_idx]
        sorted_times = journey_times[sort_idx]
        sorted_events = journey_events[sort_idx]

        # Compute NLL for the sorted journey-specific data
        journey_loss = negative_log_likelihood(sorted_events)(sorted_times, sorted_hazards)
        total_loss += journey_loss

    return total_loss


def breslow_estimator(y_train, risk_scores, eps=1e-5):
    """
    Compute the Breslow estimate of the baseline cumulative hazard.

    Parameters:
    - y_train: tuple of (Y_train, E_train)
    - risk_scores: array-like, risk scores for each observation

    Returns:
    - times: unique event times
    - H0: baseline cumulative hazard estimates at each unique event time
    """
    Y_train, E_train = y_train
    # Sort data by event times
    sorted_idx = torch.argsort(Y_train)
    sorted_times = Y_train[sorted_idx]
    sorted_indicators = E_train[sorted_idx]
    sorted_risk_scores = risk_scores[sorted_idx]

    # Unique event times
    unique_times, counts = torch.unique(sorted_times, return_counts=True)
    H0 = []
    running_risk_set = sorted_risk_scores.clone()  # Initialize the running risk set  # not exp for DSL
    
    for t, count in zip(unique_times, counts):
        dN = torch.sum(sorted_indicators[:count])
        denom = torch.sum(running_risk_set)
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
        self.batch_size = batch_size
        self.rankings = rankings
        self.full_dataset = full_dataset
        self.criterion = custom_loss_kl_pi_pitilde
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_iter = 0

    def evaluate(self, X, y, weights=None, eps=1e-5):
        self.model.eval()
        # Ensure X is a torch tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)

        # Convert weights to tensor if it's not None
        if weights is not None and isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float().to(self.device)

        # Handle y input
        if len(y) == 2:
            Y, E = y
        elif len(y) == 3:
            Y, E, J = y

        # Set batch size if not specified
        if self.batch_size is None:
            self.batch_size = len(X)

        # If input data is 2D, no batching is needed
        if len(X.shape) <= 3:
            with torch.no_grad():
                if weights is None:
                    output = torch.squeeze(self.model(X)) + eps
                else:
                    output = torch.squeeze(weights) + eps

                y_pred = torch.log(output)
                if len(y) == 2:
                    loss_function = negative_log_likelihood(E)
                    loss = loss_function(Y, y_pred)
                elif len(y) == 3:
                    loss_function = JNPLL
                    loss = loss_function(y_pred, Y, E, J)
                return loss

        # If input data is 3D or 4D, calculate loss with batching
        else:
            self.batch_size = 8
            with torch.no_grad():
                num_samples = X.shape[0]
                num_batches = (num_samples + self.batch_size - 1) // self.batch_size

                all_y_pred = torch.zeros_like(Y).to(self.device)

                for i in range(num_batches):
                    start_idx = i * self.batch_size
                    end_idx = (i + 1) * self.batch_size

                    batch_X = X[start_idx:end_idx]
                    batch_Y = Y[start_idx:end_idx]

                    if weights is None:
                        output = torch.squeeze(self.model(batch_X)) + eps
                    else:
                        batch_weights = weights[start_idx:end_idx]
                        output = torch.squeeze(batch_weights) + eps

                    batch_y_pred = torch.log(output)
                    all_y_pred[start_idx:end_idx] = batch_y_pred

                if len(y) == 2:
                    loss_function = negative_log_likelihood(E)
                    loss = loss_function(Y, all_y_pred)
                elif len(y) == 3:
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
        generator = generator.to(self.device)
        n = generator.shape[0]
        if method == "power":
            if v_init is None:
                v = torch.rand(n).to(self.device)
            else:
                v = v_init.to(self.device)

            eps = 1.0 / torch.max(torch.abs(generator))
            mat = torch.eye(n).to(self.device) + eps * generator
            A = mat.T

            v = v / torch.norm(v)
            v = torch.matmul(A, v)

            for _ in range(pepochs):
                pre_v = v
                Av = torch.matmul(A, v)
                v = Av / torch.norm(Av)
                r = v - pre_v
                normr = torch.norm(r)

                if normr < ptol:
                    break
            res = v
            return (1.0 / res.sum()) * res
        else:
            raise RuntimeError("Method not implemented")

    def ilsrx(self, y_train, rho, weights, x_beta_b, u, monitor_loss, inner_params=[100, 1e-5],
              eps=1e-5):

        iepochs = inner_params[0]
        rtol = inner_params[1]
        if len(y_train) == 2:
            Y_train, E_train = y_train
        elif len(y_train) == 3:
            Y_train, E_train, J_train = y_train

        n = len(weights)
        ilsr_conv = False
        local_iter = 0
        weights = weights.to(self.device)
        x_beta_b = x_beta_b.to(self.device)
        u = u.to(self.device)
        ilsrx_pred = torch.log(torch.squeeze(weights) + 1e-5)
        loss_function = negative_log_likelihood(E_train)
        ilsrx_loss = loss_function(Y_train, ilsrx_pred)
        monitor_loss["ilsrx_loss"].append(ilsrx_loss.item())
        self.global_iter += 1
        wandb.log({"ilsrx_loss_iter": ilsrx_loss.item(), 
                   "batch": self.global_iter}, commit=False)
        
        while not ilsr_conv:
            chain = torch.zeros((n, n), device=self.device)
            chain = MC_from_ranking(chain, self.rankings, weights, self.device, eps)

            # each row sums up to 0
            chain -= torch.diag(chain.sum(axis=1))
            weights_prev = weights.clone()
            weights = self.statdist(chain, v_init=weights)
            
            local_iter += 1
            ilsrx_pred = torch.log(torch.squeeze(weights) + 1e-5)
            loss_function = negative_log_likelihood(E_train)
            ilsrx_loss = loss_function(Y_train, ilsrx_pred)
            monitor_loss["ilsrx_loss"].append(ilsrx_loss.item())
            self.global_iter += 1
            wandb.log({"ilsrx_loss_iter": ilsrx_loss.item(), 
                   "batch": self.global_iter}, commit=False)

            # Check convergence
            ilsr_conv = torch.norm(weights_prev - weights) < rtol * torch.norm(weights) or local_iter >= iepochs
        
        return weights, chain, monitor_loss, local_iter
    
    def train_step_spectral(self, x_train, x_val, x_test, y_train, y_val, y_test, rho, monitor_loss, weights=None,
                            x_beta_b=None, u=None, gamma=1, inner_params=[100, 1e-4], avg_window=5,
                            eps=1e-5):
        
        depochs = inner_params[0]
        rtol = inner_params[1]
        
        weights, x_beta_b, u = weights.to(self.device), x_beta_b.to(self.device), u.to(self.device)
        params_conv = False
        iter = 0
        window_losses = [100000]
        nn_time = 0
        weights_prev = torch.clone(weights)
        ## pi update: no matter what the initial weights are, should come first.
        start = time.time()
        with torch.no_grad():  # Ensure no gradients flow through ilsrx
            weights, chain, monitor_loss, local_iter = self.ilsrx(y_train, rho, weights, x_beta_b, u, monitor_loss,
                                                      inner_params=inner_params)
        wandb.log({
                "conv steps": local_iter,
                "global steps": self.global_iter,
            })
        weights = weights.detach()  # Detach weights from the computation graph
        end = time.time()
        ilrsx_time = (end - start)

        MSE_score = torch.sum((x_beta_b - weights) ** 2)

        while not params_conv:
            if len(x_train.shape) > 3:
                batch_size = 4
            else:
                batch_size = self.batch_size
            
            # Training loop
            start = time.time()
            self.model.train()
            train_dataset = CustomDataset(x_train, weights, u)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            epoch_loss = 0.0

            for batch_X, batch_weights, batch_u in train_loader:
                if batch_X.size(0) == 1:
                    continue
                batch_X = batch_X.to(self.device)
                batch_weights = batch_weights.to(self.device)
                batch_u = batch_u.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(batch_X).squeeze()
                loss = self.criterion(batch_weights, y_pred, batch_u)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(train_loader.dataset)
            # Track losses
            wandb.log({
                "NNtrain_loss": epoch_loss,
            })
            end = time.time()
            nn_time += (end - start)  # log time
            window_losses.append(epoch_loss)

            iter += 1  # log number of epochs
            if iter <= avg_window:
                avg_loss = np.mean(window_losses[:-1])
            else:
                avg_loss = np.mean(window_losses[-avg_window - 1:-1])
            params_conv = (np.abs(
                avg_loss - window_losses[-1]) < 10 * rtol * avg_loss) or iter >= depochs  # check conv.

        ## dual update
        start = time.time()
        with torch.no_grad():
            x_beta_b = get_intrinsic_hazard(self.model, x_train, batch_size).squeeze()
        end = time.time()
        predict_time = (end - start)

        start = time.time()
        u += gamma * rho * (x_beta_b - weights)  # reversed
        end = time.time()
        u_time = (end - start)

        start = time.time()
        ilsrx_loss = self.evaluate(x_train, y_train, weights)
        val_loss = self.evaluate(x_val, y_val)
        train_loss = self.evaluate(x_train, y_train)
        end = time.time()
        eval_time = (end - start)
        
        if len(y_train)==2:
            test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(self.model, x_train, y_train, x_test, y_test, use_exp=False)
            val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(self.model, x_train, y_train, x_val, y_val, use_exp=False)
        else:
            # As spectral only takes unique samples
            test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(self.model, self.full_dataset[0], self.full_dataset[1], x_test, y_test, use_exp=False)
            val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(self.model, self.full_dataset[0], self.full_dataset[1], x_val, y_val, use_exp=False)
        
        print(test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE)

        # Track losses
        wandb.log({
            "ilsrx_loss": ilsrx_loss,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "MSE_score": MSE_score,
        })
        
        wandb.log({
            "test_AUC": test_AUC,
            "test_IBS": test_IBS,
            "test_CI": test_CI,
            "test_RMSE": test_RMSE, 
            "val_AUC": val_AUC,
            "val_IBS": val_IBS,
            "val-CI": val_CI, 
            "val_RMSE": val_RMSE,})

        # Track times
        wandb.log({
            "nn_time": nn_time,
            "u_time": u_time,
            "eval_time": eval_time,
            "predict_time": predict_time,
        })
        return weights_prev, x_beta_b, u, monitor_loss, chain

    def admm_kl(self, x_train, x_val, x_test, y_train, y_val, y_test, epochs,
                inner_params, patience=2):
        n = len(x_train)
        rho = 1
        weights = torch.ones(n, device=self.device) / n
        x_beta_b = weights.clone()
        u = torch.zeros(n, device=self.device)
        eps = 1e-5
        gamma = 1
        chain_list, weight_list, log = [], [], []
        monitor_loss = {}
        monitor_loss["val_loss"], monitor_loss["train_loss"], monitor_loss["ilsrx_loss"], monitor_loss["MSE_score"], \
            monitor_loss["NNtrain_loss"] = [], [], [], [], []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f'./experiment_results/model_checkpoint_{timestamp}'

        val_loss = self.evaluate(x_val, y_val)
        monitor_loss["val_loss"].append(val_loss)
        # Save initial model state
        torch.save(self.model.state_dict(), checkpoint_path)
        # pseudo_train loss
        train_loss = self.evaluate(x_train, y_train)
        monitor_loss["train_loss"].append(train_loss)
        current_loss = float('inf')
        no_improvement_count = 0

        start = time.time()
        for i in range(epochs):
            gamma /= (i + 1)
            start_spectral = time.time()
            weights, x_beta_b, u, monitor_loss, chain\
                = self.train_step_spectral(x_train, x_val, x_test, y_train, y_val, y_test, rho, monitor_loss,
                                           weights, x_beta_b, u, gamma, inner_params)
            end_spectral = time.time()
            spectral_time = (end_spectral - start_spectral)


            chain_list.append(chain)
            weight_list.append(weights)
            start_save = time.time()


            if monitor_loss["val_loss"][-1] < current_loss:
                torch.save(self.model.state_dict(), checkpoint_path)
                current_loss = monitor_loss["val_loss"][-1]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(
                    f'Early stopping at epoch {i} as validation loss did not improve for {patience} consecutive epochs.')
                break
            end_save = time.time()
            save_time = (end_save - start_save)

        end = time.time()
        admm_time = (end - start)

        start = time.time()
        # Load best model
        self.model.load_state_dict(torch.load(checkpoint_path))

        if len(y_train)==2:
            test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(self.model, x_train, y_train, x_test, y_test, use_exp=False)
            val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(self.model, x_train, y_train, x_val, y_val, use_exp=False)
        else:
            # As spectral only takes unique samples
            test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(self.model, self.full_dataset[0], self.full_dataset[1], x_test, y_test, use_exp=False)
            val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(self.model, self.full_dataset[0], self.full_dataset[1], x_val, y_val, use_exp=False)
            
        print(test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE)
        os.remove(checkpoint_path)
        
        end = time.time()
        metric_time = (end - start)
        
        wandb.log({
            "spectral_time": spectral_time,
            "save_time": save_time,
            "admm_time": admm_time,
            "metric_time": metric_time,
        })

        log = [chain_list, weight_list]

        return test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, log


    def predict_surv(self, x_train, y_train, x_test, y_test):
        X_cumhazard = self.predict_cumulative_hazards(x_train, y_train, x_test, y_test)
        survival_matrix = torch.exp(-X_cumhazard)
        return survival_matrix

    def predict_hazard(self, x):
        x = x.to(self.device)
        batch_size = self.batch_size or len(x)
        self.model.eval()
        with torch.no_grad():
            dataset = torch.tensor(x, dtype=torch.float32)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            risk_scores = []
            for batch in loader:
                scores = self.model(batch).squeeze()
                risk_scores.append(scores)
            risk_scores = torch.cat(risk_scores)
        return risk_scores

    def compute_baseline_hazards(self, x_train, y_train):
        risk_scores = self.predict_hazard(x_train)
        unique_times, H0 = breslow_estimator(y_train, risk_scores)
        return unique_times, H0

    def predict_cumulative_hazards(self, x_train, y_train, x_test, y_test):
        Y_test, E_test = y_test
        relative_hazard = self.predict_hazard(x_test)
        unique_times, H0 = self.compute_baseline_hazards(x_train, y_train)

        H0_cumulative = torch.cumsum(H0, dim=0)
        start, end = Y_test.min().item(), Y_test.max().item()
        bandwidth = (end - start) / 100
        time_grid = torch.linspace(start + 1e-5, end - 1e-5, steps=100).to(self.device)

        # Smoothed H0 with kernel smoothing
        smoothed_H0 = torch.tensor([kernel_smoothed_hazard(t, unique_times, H0_cumulative.cpu().numpy(), bandwidth)
                                    for t in time_grid.cpu().numpy()]).to(self.device)

        # Calculate cumulative hazard for the test set
        X_cumhazard = torch.matmul(relative_hazard.reshape(-1, 1), smoothed_H0.reshape(1, -1))
        return X_cumhazard


def run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout,
              batch_size, rankings, inner_params=[100, 1e-4], l2_reg=0, full_dataset=None):
    activation = 'relu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model by data
    if len(x_train.shape) < 3:
        n_features = x_train.shape[1]
        if ResNet:
            x_train = np.expand_dims(x_train, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)
            model = models.ResNet1D(l2_reg=2).to(device)  # default ResNet18
        else:
            model = models.MLPVanilla(n_features, dims, dropout=dropout).to(device)
    else:
        if ResNet:
            input_W = x_train.shape[-2]
            input_H = x_train.shape[-1]
            input_D = x_train.shape[2]
            model = generate_model(model='resnet', model_depth=depth, input_W=input_W, input_H=input_H, input_D=input_D, \
                                        resnet_shortcut='B', pretrain_path=f'./models/resnet_{depth}_23dataset.pth', sigmoid=True)
        else:
            d = x_train.shape[2]
            model = models.Conv3D_torch(depth=depth, input_dimension=d, sigmoid=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # define and train Spectral
    spectral = Spectral(model, optimizer, rankings, batch_size, full_dataset=full_dataset)
    u = torch.zeros(len(x_train), dtype=torch.float32)

    test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, log = spectral.admm_kl(
        x_train, x_val, x_test, y_train, y_val, y_test, epochs, inner_params)

    return test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, log


def run_dsl(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs, dropout,
            batch_size, rankings, inner_params=[100, 1e-4], l2_reg=2, full_dataset=None):
    return run_model(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, depth, epochs,
                     dropout, batch_size, rankings, inner_params, l2_reg, full_dataset=full_dataset)