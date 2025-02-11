import os
import pandas as pd
import time
import wandb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import requests
import zipfile

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle
import argparse

import torch.optim as optim
import torch.nn as nn

from datasets.MovieLens import load_MovieLens
from methods.utils import get_journey_rankings, get_gpu_memory_usage
from methods.preprocessing import prepare_ads
from methods.survmodel_pytorch import run_dsl
from methods.baseline import run_deepsurv

from methods.models import MLPVanilla
from methods.metrics import *

class SurvivalDataset(Dataset):
    def __init__(self, x, y):
        """
        x: Tensor of features
        y: Tuple of (times, events, journey_ids)
        """
        self.x = x
        self.times, self.events, self.journey_ids = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.times[idx], self.events[idx], self.journey_ids[idx]
    
def compute_jnpll_loss(hazards, times, events, journey_ids, device):
    unique_journeys = torch.unique(journey_ids)
    total_loss = 0.0
    for journey in unique_journeys:
        mask = journey_ids == journey
        journey_hazards = hazards[mask]
        journey_times = times[mask]
        journey_events = events[mask]

        # Sort by descending time
        sorted_indices = torch.argsort(journey_times, descending=True)
        sorted_hazards = journey_hazards[sorted_indices]
        sorted_events = journey_events[sorted_indices]

        hazard_ratio = torch.exp(sorted_hazards)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = sorted_hazards - log_risk
        censored_likelihood = uncensored_likelihood * sorted_events
        total_loss -= torch.sum(censored_likelihood)

    return total_loss / len(journey_ids)

class ModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            return self.model(x).squeeze().cpu().numpy()

    def __call__(self, x):
        return self.predict(x)
    

def download_movielens(dataset="100k", download_dir="./MovieLens"):
    """
    Downloads and extracts MovieLens data (100k, 1m, or 10m) into `download_dir`.
    
    Returns:
        ml_folder (str): Path to the extracted folder (e.g., './MovieLens/ml-100k')
    """
    # 1) Define the download URLs and folder names
    urls = {
        "100k": "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "1m":   "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "10m":  "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    }
    folders = {
        "100k": "ml-100k",
        "1m":   "ml-1m",
        "10m":  "ml-10M",
    }
    
    if dataset not in urls:
        raise ValueError(f"Invalid dataset='{dataset}'. Choose from '100k', '1m', or '10m'.")
    
    url = urls[dataset]
    folder_name = folders[dataset]
    
    os.makedirs(download_dir, exist_ok=True)
    
    zip_filename = f"{folder_name}.zip"  # e.g. 'ml-100k.zip'
    zip_path = os.path.join(download_dir, zip_filename)
    ml_folder = os.path.join(download_dir, folder_name)
    
    # 2) If folder already exists, skip download
    if os.path.isdir(ml_folder):
        return ml_folder
    
    # 3) If zip not present, download
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens {dataset} dataset...")
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)
        print("Download complete.")
    
    # 4) Extract
    print(f"Extracting {zip_filename}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(download_dir)
    print("Extraction complete.")
    
    return ml_folder


def load_and_transform_time(ml_folder, dataset="100k"):
    """        df_raw = pd.read_csv(
            ratings_file, 
            sep="\t", 
            names=["user_id", "movie_id", "rating", "timestamp"], 
            header=None
        )
    Loads the rating file from ml_folder depending on dataset type ('100k','1m','10m'),
    then transforms 'timestamp' -> days since the earliest rating in that file.
    
    Returns a DataFrame with columns: [user_id, movie_id, rating, day].
    """
    if dataset == "100k":
        ratings_file = os.path.join(ml_folder, "u.data")
        # tab-separated, 4 columns
        df_raw = pd.read_csv(
            ratings_file, 
            sep="\t", 
            names=["user_id", "movie_id", "rating", "timestamp"], 
            header=None
        )
    elif dataset in ["1m", "10m"]:
        ratings_file = os.path.join(ml_folder, "ratings.dat")
        # double-colon separated
        df_raw = pd.read_csv(
            ratings_file,
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
            header=None
        )
    else:
        raise ValueError(f"Unknown dataset='{dataset}' in load_and_transform_time.")
    
    # Convert timestamps -> days since earliest rating
    t_min = df_raw["timestamp"].min()
    df_raw["day"] = (df_raw["timestamp"] - t_min) / (60.0 * 60.0 * 24.0)
    
    return df_raw[["user_id", "movie_id", "rating", "day"]]


def build_user_movie_intervals(
    df,
    user_col="user_id",
    movie_col="movie_id",
    time_col="day",
):
    """
    For each user, if they have ratings at times = [t0, t1, ..., tN],
    we build intervals [start_i, t_i) for i=1..N, where:
        if t_i == t_(i-1),   start_i = start_(i-1)
        else               start_i = t_(i-1)

    This ensures that two movies rated at the *same timestamp* end up
    sharing the same start time.
    """
    df_sorted = df.sort_values([user_col, time_col])
    intervals = []

    for uid, group in df_sorted.groupby(user_col):
        times = group[time_col].values
        movies = group[movie_col].values

        # Edge case: if user has only 1 rating, there's nothing to build
        if len(times) < 2:
            continue

        # We'll track the "start" used for each rating
        # start[0] doesn't create an interval, it just seeds the loop.
        starts = [None] * len(times)  
        starts[0] = times[0]  # The first rating has no "previous"

        # Build intervals for ratings 1..(end)
        for i in range(1, len(times)):
            if times[i] == times[i - 1]:
                # Reuse previous rating's start if timestamps are the same
                starts[i] = starts[i - 1]
            else:
                # Normal case: start is the previous rating's time
                starts[i] = times[i - 1]

            intervals.append({
                user_col:  uid,
                movie_col: movies[i],
                "start":   starts[i],
                "stop":    times[i],
            })

    return pd.DataFrame(intervals)

def sample_censor_time_for_each_user(df_raw, user_col="user_id", 
                                     time_col="day", last_fraction=0.25, seed=42):
    """
    For each user, sample a censor time uniformly between:
    user_min + (1 - last_fraction) * (user_max - user_min) and user_max.

    Parameters:
    - df_raw (pd.DataFrame): Original DataFrame containing user events.
    - user_col (str): Column name for user IDs.
    - time_col (str): Column name for event times.
    - last_fraction (float): Fraction of the timeline to consider for censoring.
    - seed (int): Random seed for reproducibility.

    Returns:
    - user_censor_dict (dict): Mapping from user_id to T_censor^u.
    """
    rng = np.random.default_rng(seed)

    # Initialize the dictionary
    user_censor_dict = {}

    # Group by user and compute censor times individually
    for uid, group in df_raw.groupby(user_col):
        user_min = group[time_col].min()
        user_max = group[time_col].max()
        user_span = user_max - user_min
        user_censor_start = user_min + (1 - last_fraction) * user_span

        # Ensure that user_censor_start is not greater than user_max
        if user_censor_start > user_max:
            user_censor_start = user_max

        # Sample T_censor^u uniformly between user_censor_start and user_max
        T_censor_u = rng.uniform(user_censor_start, user_max)
        user_censor_dict[uid] = T_censor_u

    return user_censor_dict

def assign_events_with_user_censor(intervals_df, user_censor_dict, user_col="user_id"): 
    """
    For each interval row (start, stop):
        - If stop <= T_censor^u => event=1
        - Else:
            - If T_censor^u < start: event=0, stop=start
            - Else: event=0, stop=T_censor^u
    Calculates delta = stop - start.
    
    Returns:
        DataFrame with columns [user_id, movie_id, start, stop, event, delta].
    """
    df = intervals_df.copy()
    events = []
    new_stop = []
    
    for i, row in df.iterrows():
        uid = row[user_col]
        T_censor = user_censor_dict.get(uid, float('inf'))  # Default to infinity if not found
        
        if row["stop"] <= T_censor:
            e = 1
            s = row["stop"]
        elif T_censor < row["start"]:
            # Censor before the start of the interval
            e = 0
            s = row["start"]  # No interval duration
        else:
            # Censor within the interval
            e = 0
            s = T_censor
        
        events.append(e)
        new_stop.append(s)

    df["stop"] = new_stop
    df["event"] = events
    df["delta"] = df["stop"] - df["start"]
    
    return df


def rename_columns(df, user_col="user_id", movie_col="movie_id"):
    """
    df has columns: user_id, movie_id, delta, event
    rename => journey_id, ad_id, Y, E, and drop start/stop if desired.
    """
    df_ren = df.rename(columns={
        user_col:  "journey_id",
        movie_col: "ad_id",
        "delta":   "Y",
        "event":   "E",
    })
    # Drop start, stop to finalize
    df_ren.drop(columns=["start","stop"], inplace=True, errors="ignore")
    return df_ren

def build_movie_factors(df, user_col="user_id", movie_col="movie_id", 
                        rating_col="rating", n_factors=50):
    """
    Factorizes the user x movie rating matrix into n_factors.
    Returns a DataFrame with columns:
      [movie_id, X1..Xn_factors]
    where each row is the factor vector for one movie.
    """
    unique_users = sorted(df[user_col].unique())
    unique_movies = sorted(df[movie_col].unique())
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}

    rows = df[user_col].map(user_to_idx)
    cols = df[movie_col].map(movie_to_idx)
    vals = df[rating_col].values

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    # Build sparse rating matrix
    rating_mat = csr_matrix((vals, (rows, cols)), shape=(num_users, num_movies))

    print(f"Factorizing: rating_mat shape = {rating_mat.shape}")
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    svd.fit(rating_mat)

    # item_factors => shape: (num_movies, n_factors)
    # because svd.components_ => shape: (n_factors, num_movies)
    item_factors = svd.components_.T

    # Build DF
    factor_cols = [f"X{i}" for i in range(1, n_factors+1)]
    movie_factors_df = pd.DataFrame(item_factors, columns=factor_cols)
    movie_factors_df[movie_col] = unique_movies

    return movie_factors_df


def split_by_ad_id(df, ad_col="ad_id", train_ratio=0.7, val_ratio=0.15, 
                   test_ratio=0.15, min_ratings=0, random_state=42):
    """
    Splits df so that train/val/test have disjoint sets of ad_id (movie_id).
    """
    # Filter out users with fewer than min_ratings
    user_counts = df["journey_id"].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    df = df[df["journey_id"].isin(valid_users)].copy()
    df = df[df['Y'] != 0]
    
    unique_ads = sorted(df[ad_col].unique())
    n_ads = len(unique_ads)
    n_train = int(train_ratio * n_ads)
    n_val   = int(val_ratio   * n_ads)

    train_ads = unique_ads[:n_train]
    val_ads   = unique_ads[n_train : n_train + n_val]
    test_ads  = unique_ads[n_train + n_val :]

    df_train = df[df[ad_col].isin(train_ads)].copy()
    df_val   = df[df[ad_col].isin(val_ads)].copy()
    df_test  = df[df[ad_col].isin(test_ads)].copy()
    
    # Remap ad_id in df_train to be continuous integers starting from 1
    unique_train_ads = sorted(df_train[ad_col].unique())
    train_ad_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_train_ads, start=1)}
    df_train[ad_col] = df_train[ad_col].map(train_ad_id_mapping)

    return df_train, df_val, df_test


def load_MovieLens(dataset="100k", min_ratings=200):
    """
    Complete pipeline to process MovieLens data with pickle caching.
    If the processed pickle file exists, load and return it.
    Else, process the data, save to pickle, and return.
    
    Args:
        dataset (str): '100k', '1m', or '10m'.
    
    Returns:
        dfs (dict): Dictionary containing train, val, and test DataFrames.
    """
    # Define file name for pickle
    file_name = f"./datasets/MovieLens/MovieLens{dataset}.pkl"
    
    # Check if pickle exists
    if os.path.exists(file_name):
        print(f"Loading existing pickle file: {file_name}")
        with open(file_name, 'rb') as f:
            dfs = pickle.load(f)
        print("Data loaded from pickle.")
        return dfs
    
    # Else, proceed with processing
    print(f"Pickle file not found. Processing dataset '{dataset}'...")
    
    # 1) Download
    ml_folder = download_movielens(dataset=dataset, download_dir="./MovieLens")
    
    # 2) Load + transform time -> days
    df_raw = load_and_transform_time(ml_folder, dataset=dataset)
    # df_raw => [user_id, movie_id, rating, day]
    print("Raw DataFrame (first 5 rows):")
    print(df_raw.head(5))
    
    # 3) Build user-based intervals, keeping movie_id_start
    intervals_df = build_user_movie_intervals(
        df_raw, 
        user_col="user_id", 
        time_col="day", 
        movie_col="movie_id"
    )
    
    if intervals_df.empty:
        print("No intervals were created. Users may have only one rating each.")
        return {}
    print("Intervals:")
    print(intervals_df.head(5))
    # 3.a) Sample censor times for each user from the last 25% of the global time range
    global_min = df_raw["day"].min()
    global_max = df_raw["day"].max()
    
    user_censor_dict = sample_censor_time_for_each_user(
        df_raw, 
        last_fraction=0.25, 
        seed=42
    )
    
    # 3.b) Assign events based on censor times
    censored_df = assign_events_with_user_censor(
        intervals_df, 
        user_censor_dict, 
        user_col="user_id"
    )
    print("\nCensored Intervals DataFrame (first 5 rows):")
    print(censored_df.head(5))
    

    
    # 4) Matrix factorization to get 50 movie features
    movie_factors_df = build_movie_factors(
        df_raw, 
        user_col="user_id", 
        movie_col="movie_id", 
        rating_col="rating", 
        n_factors=50
    )
    print("\nMovie Factors DataFrame (first 5 rows):")
    print(movie_factors_df.head(5))
    
    # 5) Merge intervals with movie factors
    merged_df = pd.merge(
        censored_df, 
        movie_factors_df, 
        on="movie_id", 
        how="left"
    )
    print("\nMerged DataFrame (first 5 rows):")
    print(merged_df.head(5))
    
    # 6) Rename columns
    renamed_df = rename_columns(
        merged_df, 
        user_col="user_id", 
        movie_col="movie_id"
    )
    print("\nRenamed DataFrame (first 5 rows):")
    print(renamed_df.tail(5))
    
    # 7) Stack into final DataFrame
    factor_cols = [f"X{i}" for i in range(1, 51)]
    final_cols = ["journey_id", "ad_id", "E", "Y"] + factor_cols
    df_final = renamed_df[final_cols].sort_values("ad_id").reset_index(drop=True)
    
    print("\nFinal DataFrame Shape:", df_final.shape)
    print("Final DataFrame (first 10 rows):")
    print(df_final.tail(10))

    # 8) Split into train, val, test
    df_train, df_val, df_test = split_by_ad_id(
        df_final, 
        ad_col="ad_id",
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15, 
        min_ratings=min_ratings,
        random_state=42
    )
    print(f"\nTrain Set: {df_train.shape}")
    print(f"Validation Set: {df_val.shape}")
    print(f"Test Set: {df_test.shape}")
    print("Unique ad_id counts:")
    print(f"Train: {df_train['ad_id'].nunique()}, Val: {df_val['ad_id'].nunique()}, Test: {df_test['ad_id'].nunique()}")
    
    # 9) Save splits to pickle
    dfs = {'train': df_train, 'val': df_val, 'test': df_test}
    
    directory = os.path.dirname(file_name)

    # Create the directory if it doesn't exist
    if directory:  # This check ensures that if file_name is just a filename without a directory, it won't attempt to create an empty path
        os.makedirs(directory, exist_ok=True)
    
    with open(file_name, 'wb') as f:
        pickle.dump(dfs, f)
    print(f"\nData splits saved to {file_name}")
    
    return dfs


def run_deepsurv_movie(
    ResNet,
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
    learning_rate,
    dims,
    depth,
    epochs,
    dropout,
    batch_size,
    batch_norm=False,
    out_features=1,
    output_bias=False,
    patience=10
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare datasets and dataloaders
    datasets = {
        'train': SurvivalDataset(x_train, y_train),
        'val': SurvivalDataset(x_val, y_val),
        'test': SurvivalDataset(x_test, y_test)
    }
    print(f"BS:{batch_size}")
    loaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k == 'train'))
        for k, v in datasets.items()
    }

    # Initialize the network
    if ResNet:
        raise NotImplementedError("ResNet architecture is not implemented in this version.")
    else:
        in_features = x_train.shape[1]
        model = MLPVanilla(
            in_features=in_features,
            num_nodes=dims,
            out_features=out_features,
            batch_norm=batch_norm,
            dropout=dropout,
            activation=nn.ReLU,
            output_activation=None,  # Typically, no activation for Cox model
            output_bias=output_bias
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss, epochs_no_improve = float('inf'), 0
    best_state = None
    log_history = []

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_times, batch_events, batch_journey_ids in loaders['train']:
            batch_x = batch_x.to(device)
            batch_times = batch_times.to(device)
            batch_events = batch_events.to(device)
            batch_journey_ids = batch_journey_ids.to(device)

            optimizer.zero_grad()
            hazards = model(batch_x).squeeze()
            loss = compute_jnpll_loss(hazards, batch_times, batch_events, batch_journey_ids, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_train_loss = epoch_loss / len(loaders['train'].dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_times, batch_events, batch_journey_ids in loaders['val']:
                print(f"batch x time event journey shape:{batch_x.shape, batch_times.shape, batch_events.shape, len(batch_journey_ids)}")
                batch_x = batch_x.to(device)
                batch_times = batch_times.to(device)
                batch_events = batch_events.to(device)
                batch_journey_ids = batch_journey_ids.to(device)

                hazards = model(batch_x).squeeze()
                loss = compute_jnpll_loss(hazards, batch_times, batch_events, batch_journey_ids, device)
                val_loss += loss.item() * batch_x.size(0)

        avg_val_loss = val_loss / len(loaders['val'].dataset)
        log_history.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Wrap the model for sksurv_metrics
        wrapped_model = ModelWrapper(model, device)

        
        val_AUC, val_IBS, val_CI, val_RMSE = sksurv_metrics(
        model=wrapped_model,
        x_train=x_train.cpu().numpy(),
        y_train=y_train,
        x_test=x_val.cpu().numpy(),
        y_test=y_val,
        model_name="DeepSurv"
        )

        test_AUC, test_IBS, test_CI, test_RMSE = sksurv_metrics(
            model=wrapped_model,
            x_train=x_train.cpu().numpy(),
            y_train=y_train,
            x_test=x_test.cpu().numpy(),
            y_test=y_test,
            model_name="DeepSurv"
        )

        wandb.log({
            "test_AUC": test_AUC,
            "test_IBS": test_IBS,
            "test_CI": test_CI,
            "test_RMSE": test_RMSE, 
            "val_AUC": val_AUC,
            "val_IBS": val_IBS,
            "val-CI": val_CI, 
            "val_RMSE": val_RMSE,})
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # Load the best model
    if best_state:
        model.load_state_dict(best_state)

    return test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, log_history



def get_journey_rankings(df_train, min_id=0):
    Y = df_train['Y'].values
    E = df_train['E'].values
    J = df_train['journey_id'].values
    A= df_train['ad_id'].values

    rankings = []
    unique_J = np.unique(J)
    for journey_id in unique_J:
        # Get the indices of the current journey_id
        journey_indices = np.where(J == journey_id)[0]
        # Get the corresponding ad_id values for the current journey_id
        journey_ad_ids = A[journey_indices]
        journey_Y = Y[journey_indices]
        if min_id == 0:
            ranking = journey_ad_ids[np.argsort(journey_Y)]
        else:
            ranking = journey_ad_ids[np.argsort(journey_Y)]-1
        rankings.append(ranking)
    return rankings


def parse_arguments():
    parser = argparse.ArgumentParser(description="MovieLens Survival Analysis")
    parser.add_argument('--dataset', type=str, default="MovieLens100k", 
    choices=["Ads100", "Ads1k", "Ads10k", "MovieLens100k"], 
    help="Dataset choice: Ads100, Ads1k, Ads10k for synthetic ads datasets, or MovieLens100k for MovieLens.")
    parser.add_argument('--resnet', action='store_true', help="Use ResNet architecture")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--depth', type=int, default=6, help="Network depth")
    parser.add_argument('--dims', type=int, nargs='+', default=[200], help="Dimensions for hidden layers")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--batch_size_coef', type=float, default=0.01, help="Batch size coefficient")
    parser.add_argument("--algorithm", nargs='?', action="store", type=str, help='spectral or baseline', default="spectral")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--rtol',  nargs='?',help='rtol', type=float, default=1e-4)
    parser.add_argument('--n_iter',  nargs='?',help='rtol', type=int, default=100)
    parser.add_argument('--reg',  nargs='?',help='reg', type=float, default=16)

    return parser.parse_args()

def main():
    args = parse_arguments()
    start_time = time.time()
    dataset = args.dataset
    # Load and preprocess data
    if dataset.startswith("Ads"):
        n_map = {"Ads100": 100, "Ads1k": 1000, "Ads10k": 10000}
        n = n_map.get(dataset, None)

        if n is None:
            raise ValueError(f"Unknown dataset name: {dataset}")

        m, s, feature_dim = 50, 200, 50  # Adjust these values as needed
        file_name = f'./datasets/Synthetic/synthetic_n{n}_m{m}_s{s}_dim{feature_dim}.pickle'

        df = pd.read_pickle(file_name) 
        # The train val test are split by ads, so we cannot use CV here
        df_train, df_val, df_test = df["train"], df["val"], df["test"]
        rankings = get_journey_rankings(df_train, min_id=0)
    
    else:
        dfs = load_MovieLens(dataset="100k", min_ratings=0)
        df_train, df_val, df_test = dfs["train"], dfs["val"], dfs["test"]
        
        rankings = get_journey_rankings(df_train, min_id=1)
        
    x_train, x_train_part, x_val, x_test, y_train, y_train_part, y_val, y_test = prepare_ads(df_train, df_val, df_test, tensor=True)
    print("num of training samples:", x_train.shape, "part of training", x_train_part.shape)
    
    # Model configuration
    dims = tuple(args.dims * (args.depth - 1))
    batch_size = round(len(x_train) * args.batch_size_coef)
    print(f"num_train:{len(x_train)}--BS:{batch_size}--batch_size_coef{args.batch_size_coef}")
    config = {
        "dataset": args.dataset,
        "algorithm": args.algorithm,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "dims": dims,
        "depth": args.depth,
        "dropout": args.dropout,
        "batch_size_coef": args.batch_size_coef,
        "ResNet": args.resnet,
    }
    print(f"Selected dataset: {args.dataset}")
    
    # Model training
    if args.algorithm == "deepsurv":
        wandb.init(project=f"MovieLens{args.dataset}_cv", name=f"DeepSurv_Res{args.resnet}_{args.depth}_{args.learning_rate}_{args.batch_size_coef}", config=config)
        # Use cross_validate_baseline for baseline algorithms
        test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, log\
                        = run_deepsurv_movie(args.resnet, x_train, x_val, x_test, y_train, y_val, y_test,
                                            args.learning_rate, dims, args.depth, args.epochs, args.dropout,
                                            batch_size, batch_norm=False, out_features=1, output_bias=False
                                                )
        print(test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE)
    else:
        wandb.init(project=f"MovieLens{args.dataset}_cv", name=f"Spectral_{args.depth}_{args.learning_rate}_{args.batch_size_coef}_{args.n_iter}_{args.rtol}", config=config)
        test_AUC, test_IBS, test_CI, test_RMSE, val_AUC, val_IBS, val_CI, val_RMSE, log = run_dsl(
            args.resnet, x_train_part, x_val, x_test, y_train_part, y_val, y_test,
            args.learning_rate, dims, args.depth, args.epochs, args.dropout,
            batch_size, rankings, inner_params=[args.n_iter, args.rtol], l2_reg=args.reg, full_dataset=(x_train, y_train)
        )
    
    elapsed_time = time.time() - start_time
    used_memory, total_memory = get_gpu_memory_usage(0)
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"GPU Memory Usage: {used_memory} MB / {total_memory} MB")
    wandb.log({
        "time": elapsed_time,
        "Memory": used_memory,
    })

if __name__ == "__main__":
    main()