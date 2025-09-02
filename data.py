import os
import torch
import pandas as pd
import numpy as np
device = torch.device("cpu")
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_data(file_path='radiation_data_xyzri_phi10.csv', processed_file='processed_data.pt'):
    """
    Loads and preprocesses the dataset, ensuring compatibility with model training.
    """
    if os.path.exists(processed_file):
        file_mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
        processed_mtime = os.path.getmtime(processed_file)
        if file_mtime < processed_mtime:
            print(f"Loading preprocessed data from: {processed_file}")
            try:
                data_dict = torch.load(processed_file, weights_only=False)
                coords = data_dict['coords'].to(device)
                I_true = data_dict['I_true'].to(device)
                I_true_normalized = data_dict['I_true_normalized'].to(device)
                return (coords, I_true, I_true_normalized,
                        data_dict['ri_mean'], data_dict['ri_std'],
                        data_dict['ri_mean_original'], data_dict['ri_std_original'])
            except Exception as e:
                print(f"Failed to load preprocessed file: {str(e)}. Regenerating...")
                os.remove(processed_file)

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None, None, None, None, None, None, None

    try:
        print("Attempting to read CSV with comma delimiter...")
        data = pd.read_csv(file_path)
    except pd.errors.ParserError:
        print("Comma delimiter failed. Trying semicolon...")
        data = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Failed to read CSV: {str(e)}")
        return None, None, None, None, None, None, None

    print("CSV columns:", data.columns.tolist())
    data.columns = data.columns.str.strip()

    expected_columns = ['x', 'y', 'z', 'ri']
    if not set(expected_columns).issubset(data.columns):
        print("Column names do not match. Trying common alternatives...")
        possible_columns = {'intensity': 'ri', 'I': 'ri', 'radiative_intensity': 'ri'}
        for old_col, new_col in possible_columns.items():
            if old_col in data.columns:
                data.rename(columns={old_col: new_col}, inplace=True)
        if not set(expected_columns).issubset(data.columns):
            print(f"Required columns {expected_columns} not found. Available: {data.columns.tolist()}")
            intensity_column = input("Please enter the correct column name for intensity: ")
            if intensity_column in data.columns:
                data.rename(columns={intensity_column: 'ri'}, inplace=True)
            else:
                print(f"Column {intensity_column} not found. Aborting.")
                return None, None, None, None, None, None, None

    print("\nFirst few rows of data:\n", data.head())
    print("\nData info:\n", data.info())
    print("\nData statistics:\n", data.describe())

    data['ri'].fillna(data['ri'].mean(), inplace=True)

    ri_mean_original = data['ri'].mean()
    ri_std_original = data['ri'].std()

    ri_mean = data['ri'].mean()
    ri_std = data['ri'].std()
    outlier_threshold = 3
    outliers = (data['ri'] < (ri_mean - outlier_threshold * ri_std)) | (data['ri'] > (ri_mean + outlier_threshold * ri_std))
    print(f"\nDetected {outliers.sum()} outliers. Replacing with mean...")
    data.loc[outliers, 'ri'] = ri_mean

    ri_mean = data['ri'].mean()
    ri_std = data['ri'].std()
    data['ri'] = (data['ri'] - ri_mean) / ri_std

    print("\nNormalized data statistics:\n", data.describe())

    coords = torch.tensor(data[['x', 'y', 'z']].values, dtype=torch.float32).to(device)
    I_true = torch.tensor(data['ri'].values, dtype=torch.float32).to(device)
    I_true_normalized = I_true

    print("\nData loaded successfully!")
    print("\nCoordinate data (first five rows):\n", data[['x', 'y', 'z']].head())
    print("\nRadiation intensity (first five rows):\n", data['ri'].head())

    data_dict = {
        'coords': coords.cpu(),
        'I_true': I_true.cpu(),
        'I_true_normalized': I_true_normalized.cpu(),
        'ri_mean': ri_mean,
        'ri_std': ri_std,
        'ri_mean_original': ri_mean_original,
        'ri_std_original': ri_std_original
    }
    torch.save(data_dict, processed_file)
    print(f"Preprocessed data saved to: {processed_file}")

    return coords, I_true, I_true_normalized, ri_mean, ri_std, ri_mean_original, ri_std_original