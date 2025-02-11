from SurvSet.data import SurvLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import train_test_split=
import numpy as np
from sklearn_pandas import DataFrameMapper
from datasets import metabric, support, whas
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
import torchtuples as tt
import pickle
import torch
import os
import torch.nn.functional as F


def load_BraTS(tensor=False, img_size=128):
    with open("./datasets/BraTS2020.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)

    x_train = loaded_dataset["x_train"]
    x_test = loaded_dataset["x_test"]
    
    # Convert the list of NumPy arrays to a single NumPy array
    y_train = loaded_dataset["y_train"]
    y_test = loaded_dataset["y_test"]
    
    if tensor:
        x_train = torch.tensor(np.array(loaded_dataset["x_train"])).permute(0, 3, 1, 2).unsqueeze(1).float()
        # x_val = torch.tensor(loaded_dataset["X_val"]).permute(0, 3, 1, 2).unsqueeze(1)
        x_test = torch.tensor(np.array(loaded_dataset["x_test"])).permute(0, 3, 1, 2).unsqueeze(1).float()
        
        # Resize the height and width to img_size
        if img_size is not None:
            x_train = F.interpolate(x_train, size=(x_train.shape[2], img_size, img_size), mode='trilinear', align_corners=False)
            x_test = F.interpolate(x_test, size=(x_test.shape[2], img_size, img_size), mode='trilinear', align_corners=False)
        
        # Convert the NumPy arrays to PyTorch tensors
        Y_train, E_train = torch.tensor(y_train)
        Y_test, E_test = torch.tensor(y_test)
        y_train,  y_test = (Y_train, E_train), (Y_test, E_test)
     
    return x_train, x_test, y_train, y_test

def load_lung1(tensor=False):
    with open("./datasets/lung1.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)

    x_train = loaded_dataset["X_train"]
    x_test = loaded_dataset["X_test"]
    
    # Convert the list of NumPy arrays to a single NumPy array
    y_train = loaded_dataset["y_train"]
    y_test = loaded_dataset["y_test"]
    
    if tensor:
        x_train = torch.tensor(np.array(loaded_dataset["X_train"])).permute(0, 3, 1, 2).unsqueeze(1)
        x_test = torch.tensor(np.array(loaded_dataset["X_test"])).permute(0, 3, 1, 2).unsqueeze(1)

        Y_train, E_train = map(lambda x: torch.from_numpy(np.array(x)) if isinstance(x, (list, np.ndarray)) else x, y_train)
        Y_test, E_test = map(lambda x: torch.from_numpy(np.array(x)) if isinstance(x, (list, np.ndarray)) else x, y_test)
        y_train,  y_test = (Y_train, E_train), (Y_test, E_test)
     
    return x_train, x_test, y_train, y_test


def hazard_rate(ad_feature, true_coef, noise):
    # assume exponential distribution
    baseline_hazard = 1 + noise
    return np.exp(np.dot(ad_feature, true_coef)) * baseline_hazard

def generate_survival_time(ad_feature, true_coef, noise):
    u = np.random.uniform()
    lambda_ = hazard_rate(ad_feature, true_coef, noise)
    return -np.log(u) / lambda_

def generate_synthetic_data(n, m, s, feature_dim, true_coef, scale=0.01, censor_prob=0.5):
    # Proportional AD split for training (70%), validation (15%), and test (15%) sets
    np.random.seed(0)  # For reproducibility
    
    num_train_ads = int(s * 0.7)
    num_val_ads = int(s * 0.15)
    num_test_ads = s - num_train_ads - num_val_ads
    
    ad_indices = np.arange(s)
    train_ads = ad_indices[:num_train_ads]
    val_ads = ad_indices[num_train_ads:num_train_ads + num_val_ads]
    test_ads = ad_indices[num_train_ads + num_val_ads:]
    
    # Generate ad features
    X = 0.1 * np.random.randn(s, feature_dim)
    
    # Create containers for the synthetic data
    all_data = {
        'train': {'journey_ids': [], 'ad_ids': [], 'Y': [], 'E': [], 'features': []},
        'val': {'journey_ids': [], 'ad_ids': [], 'Y': [], 'E': [], 'features': []},
        'test': {'journey_ids': [], 'ad_ids': [], 'Y': [], 'E': [], 'features': []}
    }
    
    for journey_id in range(n+200):
        # Determine which set this journey belongs to (train, val, test)
        if journey_id < n:  # n training
            set_type = 'train'
            m_train = min(m, len(train_ads))
            num_train = np.random.randint(1, m_train + 1)
            picked_ads = np.random.choice(train_ads, num_train, replace=False)
        elif n <= journey_id < n +100:  # 100 journeys validation
            set_type = 'val'
            m_val = min(m, len(val_ads))
            num_val = np.random.randint(1, m_val + 1)
            picked_ads = np.random.choice(val_ads, num_val, replace=False)
        else:  # 100 journeys test
            set_type = 'test'
            m_test = min(m, len(test_ads))
            num_test = np.random.randint(1, m_test + 1)
            picked_ads = np.random.choice(test_ads, num_test, replace=False)
        
        X_picked = X[picked_ads]
        
        # Generate survival times and censoring for each picked ad
        survival_times = []
        event_observed = []
        
        for ad_feature, ad_id in zip(X_picked, picked_ads):
            event = np.random.binomial(1, 1 - censor_prob)
            event_observed.append(event)
            noise = np.random.normal(scale=scale)
            survival_time = generate_survival_time(ad_feature, true_coef, noise)
            if event == 1:
                observed_time = survival_time
            else:
                observed_time = np.random.uniform(0, survival_time) # censored by some censoring time before surv_time
            survival_times.append(survival_time)
            
            
            all_data[set_type]['ad_ids'].append(ad_id)
        
        # Append the results for this journey to the containers
        all_data[set_type]['journey_ids'].extend([journey_id] * len(picked_ads))
        all_data[set_type]['Y'].extend(survival_times)
        all_data[set_type]['E'].extend(event_observed)
        all_data[set_type]['features'].extend(X_picked)
    
    # Create DataFrames for training, validation, and test sets
    df_train = pd.DataFrame({
        'journey_id': all_data['train']['journey_ids'],
        'ad_id': all_data['train']['ad_ids'],
        'E': all_data['train']['E'],
        'Y': all_data['train']['Y'],
    })
    df_val = pd.DataFrame({
        'journey_id': all_data['val']['journey_ids'],
        'ad_id': all_data['val']['ad_ids'],
        'E': all_data['val']['E'],
        'Y': all_data['val']['Y'],
    })
    df_test = pd.DataFrame({
        'journey_id': all_data['test']['journey_ids'],
        'ad_id': all_data['test']['ad_ids'],
        'E': all_data['test']['E'],
        'Y': all_data['test']['Y'],
    })
    
    for i in range(feature_dim):
        df_train[f'X{i+1}'] = np.array(all_data['train']['features'])[:, i]
        df_val[f'X{i+1}'] = np.array(all_data['val']['features'])[:, i]
        df_test[f'X{i+1}'] = np.array(all_data['test']['features'])[:, i]
     # Find the maximum survival time in the training set
    max_train_survival_time = df_train['Y'].max()
    
    # Filter out samples in val and test sets that have survival times longer than max_train_survival_time
    df_val = df_val[df_val['Y'] <= max_train_survival_time].reset_index(drop=True)
    df_test = df_test[df_test['Y'] <= max_train_survival_time].reset_index(drop=True)
    
    return df_train, df_val, df_test


def load_syn_ads(n=1000, m=50, s=200, feature_dim=50):
    file_name = f'./datasets/Synthetic/synthetic_n{n}_m{m}_s{s}_dim{feature_dim}.pickle'
    if os.path.isfile(file_name):
        print('loading synthetic.pickle')
        with open(file_name, 'rb') as f:
            loaded_dfs = pickle.load(f)
            df_train, df_val, df_test= loaded_dfs['train'], loaded_dfs['val'],\
                                                loaded_dfs['test']
    else:
        true_coef = np.random.randn(feature_dim)
        df_train, df_val, df_test = generate_synthetic_data(n, m, s, feature_dim, true_coef)
        dfs = {'train': df_train, 'val': df_val, 'test': df_test}
        with open(file_name, 'wb') as f:
            pickle.dump(dfs, f)
    return df_train, df_val, df_test


def prepare_ads(df_train, df_val, df_test, n=None, tensor=False):
    """
    Extracts and stacks the specified columns ('Y', 'E', and 'journey_id') into tuples for survival analysis
    from training, validation, and test sets. Converts the data to PyTorch tensors and moves them to CUDA.

    Parameters:
    df_train (pd.DataFrame): The training set dataframe.
    df_val (pd.DataFrame): The validation set dataframe.
    df_test (pd.DataFrame): The test set dataframe.
    n (int, optional): Number of samples to use from the training set. Defaults to the size of df_train.

    Returns:
    tuple: (x_train_full, x_train, x_val, x_test, y_train_full, y_train, y_val, y_test)
        x_train_full, x_train, x_val, x_test: PyTorch tensors containing the feature columns for training, validation, and test sets.
        y_train_full, y_train, y_val, y_test: Tuples of (times, events, journey_ids) as PyTorch tensors.
    """
    # Set device to CUDA if available, else CPU
    if tensor:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if n is None:
        n = df_train.shape[0]
    
    # ============================
    # Process Training Data (Unique Ads)
    # ============================
    df_unique_ad = df_train.drop_duplicates(subset='ad_id').copy()
    df_unique_ad = df_unique_ad.sort_values(by='ad_id').reset_index(drop=True)
    
    # Features
    x_train = df_unique_ad.drop(columns=['Y', 'E', 'journey_id', 'ad_id'])
    x_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    
    # Targets
    y_train_times = torch.tensor(df_unique_ad['Y'].values, dtype=torch.float32).to(device)
    y_train_events = torch.tensor(df_unique_ad['E'].values, dtype=torch.float32).to(device)
    y_train_journey_ids = torch.tensor(df_unique_ad['journey_id'].values, dtype=torch.long).to(device)
    y_train = (y_train_times, y_train_events, y_train_journey_ids)
    
    # ============================
    # Process Full Training Data
    # ============================
    x_train_full = df_train.drop(columns=['Y', 'E', 'journey_id', 'ad_id'])
    x_train_full = torch.tensor(x_train_full.values, dtype=torch.float32).to(device)
    
    y_train_full_times = torch.tensor(df_train['Y'].values, dtype=torch.float32).to(device)
    y_train_full_events = torch.tensor(df_train['E'].values, dtype=torch.float32).to(device)
    y_train_full_journey_ids = torch.tensor(df_train['journey_id'].values, dtype=torch.long).to(device)
    y_train_full = (y_train_full_times, y_train_full_events, y_train_full_journey_ids)
    
    # ============================
    # Process Validation Data
    # ============================
    x_val = df_val.drop(columns=['Y', 'E', 'journey_id', 'ad_id'])
    x_val = torch.tensor(x_val.values, dtype=torch.float32).to(device)
    
    y_val_times = torch.tensor(df_val['Y'].values, dtype=torch.float32).to(device)
    y_val_events = torch.tensor(df_val['E'].values, dtype=torch.float32).to(device)
    y_val_journey_ids = torch.tensor(df_val['journey_id'].values, dtype=torch.long).to(device)
    y_val = (y_val_times, y_val_events, y_val_journey_ids)
    
    # ============================
    # Process Test Data
    # ============================
    x_test = df_test.drop(columns=['Y', 'E', 'journey_id', 'ad_id'])
    x_test = torch.tensor(x_test.values, dtype=torch.float32).to(device)
    
    y_test_times = torch.tensor(df_test['Y'].values, dtype=torch.float32).to(device)
    y_test_events = torch.tensor(df_test['E'].values, dtype=torch.float32).to(device)
    y_test_journey_ids = torch.tensor(df_test['journey_id'].values, dtype=torch.long).to(device)
    y_test = (y_test_times, y_test_events, y_test_journey_ids)
    
    return x_train_full, x_train, x_val, x_test, y_train_full, y_train, y_val, y_test



def get_journey_rankings(df_train):
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
        ranking = journey_ad_ids[np.argsort(journey_Y)]
        rankings.append(ranking)
    return rankings


def sort_data(X, Y, E, C=None):
    sort_idx = np.argsort(Y)[::-1]
    X, Y, E = X[sort_idx], Y[sort_idx], E[sort_idx]
    if C is None:
        return X, Y, E
    else:
        C = C[sort_idx]
        return X, Y, E, C


def check_data(X_train, Y_train, E_train, X_val, Y_val, E_val):
    if Y_val.max() >= Y_train.max():
        move_idx = Y_val.argsort()[-1]
        move_X, move_Y, move_E = X_val[move_idx], Y_val[move_idx], E_val[move_idx]
        X_train = np.vstack([move_X, X_train])
        Y_train = np.hstack([np.array([move_Y]), Y_train])
        E_train = np.hstack([move_E, E_train])
        X_val = np.delete(X_val, move_idx, axis=0)
        Y_val = np.delete(Y_val, move_idx, axis=0)
        E_val = np.delete(E_val, move_idx, axis=0)
    return X_train, Y_train, E_train, X_val, Y_val, E_val


def prep_data(dataset, frac=0.1):
    # outside sources like DLBCL, VDV, BDCD, LUAD
    if dataset == "whas":
        df_train = whas.read_df()
    elif dataset == "metabric":
        df_train = metabric.read_df()
    elif dataset == "support":
        df_train = support.read_df()
    else:
        loader = SurvLoader()
        df_train, ref = loader.load_dataset(ds_name=dataset).values()
        df_train.rename(columns={'time':'duration'}, inplace=True)
        df_train = df_train.drop(df_train.columns[0], axis=1)
        # Assuming df_train is your DataFrame
        first_columns = pd.concat([df_train.iloc[:, 1], df_train.iloc[:, 0]], axis=1) # Select the first two columns
        remaining_columns = df_train.iloc[:, 2:]  # Select the columns excluding the first two
        
        df_train = pd.concat([remaining_columns, first_columns], axis=1)
        
    df_test = df_train.sample(frac=frac)
    df_train = df_train.drop(df_test.index)
    return df_train, df_test


def preprocessing(dataset, df_train, df_val, df_test, classed=False):
    if dataset == "metabric":
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')

    elif dataset == "support":
        cols_standardize =  ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
        cols_leave = ['x1', 'x4', 'x5']
        cols_categorical =  ['x2', 'x3', 'x6']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

        x_mapper_float = DataFrameMapper(standardize + leave)
        x_mapper_long = DataFrameMapper(categorical)  # we need a separate mapper to ensure the data type 'int64'
        x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df), x_mapper_long.fit_transform(df))
        x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df), x_mapper_long.transform(df))
        x_train = x_fit_transform(df_train)
        x_val = x_transform(df_val)
        x_test = x_transform(df_test)
        x_train = x_train[0]
        x_val = x_val[0]
        x_test = x_test[0]
    # All cols
    else:
        cols_list = df_train.columns.tolist()
        all_cols = cols_list[:cols_list.index('duration')]
        standardize = [([col], StandardScaler()) for col in all_cols]
        x_mapper = DataFrameMapper(standardize)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')
        
    n_patients_train = x_train.shape[0]
    n_features = x_train.shape[1]
    
    get_target = lambda df: (df['duration'].values, df['event'].values)
    Y_train, E_train = get_target(df_train)
    Y_val, E_val = get_target(df_val)
    Y_test, E_test = get_target(df_test)
    # Sort your training data based on Y_train
    X_train, Y_train, E_train = sort_data(x_train, Y_train, E_train)
    X_val, Y_val, E_val = sort_data(x_val, Y_val, E_val)
    X_test, Y_test, E_test = sort_data(x_test, Y_test, E_test)
    X_train, Y_train, E_train, X_val, Y_val, E_val = check_data(X_train, Y_train, E_train, X_val, Y_val, E_val)
    X_train, Y_train, E_train, X_test, Y_test, E_test = check_data(X_train, Y_train, E_train, X_test, Y_test, E_test)
    ranking = np.argsort(Y_train.squeeze())[::1]
    return  X_train, X_val, X_test, (Y_train, E_train), (Y_val, E_val), (Y_test, E_test)#, n_patients_train, n_features, ranking