import os
import glob
import pkg_resources
import pandas as pd
import numpy as np
import random
import re
from scipy import signal, stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

def set_seeds(seed=42):
    '''
    Function to set the various random seeds

    Input:
        seed (int): The seed to set
    '''
    # set the pytorch CPU seed
    torch.manual_seed(seed)
    
    # set the pytorch MPS seed
    torch.mps.manual_seed(seed)

    # set numpy seed
    np.random.seed(seed)

    # set python's built-in random seed
    random.seed(seed)

"""
Dataset class that extracts statistical features from raw sensor data and sets up features with the metadata

Class Members:
    X: Raw sensor data for each step
    y: Labels for each step (binary: MS/non-MS)
    session_ids: Session identifier for each step
    metadata: DataFrame containing metadata like duration for each step
    window_size: Number of consecutive steps to include in each window (default: 4)
"""
class SensorDataset(Dataset):
    def __init__(self, X, y, session_ids, metadata, window_size=4):
        self.metadata = metadata
        self.session_ids = session_ids
        self.window_size = window_size

        # extract statistical features for all windows, then append duration feature
        self.X = []
        self.y = []
        self.X, self.y = self.extract_features(X, y, session_ids, window_size)
    
    def extract_features(self, X, y, session_ids, window_size):
        """
        Extract statistical features from each window of sensor data.
        
        Returns:
            - an array of statistical features and an array of labels
        """
        # group data by session to ensure windows don't span across different sessions
        session_groups = {}
        for idx, session_id in enumerate(session_ids):
            session_groups.setdefault(session_id, []).append((X[idx], y[idx]))
        
        features = []
        labels = []
        # create sliding windows within each session
        for session_id, session_data in session_groups.items():
            # get the metadata for this session sorted by start time
            session_meta = self.metadata[self.metadata['session_id'] == session_id].sort_values(by='start').reset_index(drop=True)
            
            for i in range(len(session_data) - window_size + 1):
                # get the raw sensor data for each step in this window
                steps = [data[0] for data in session_data[i:i+window_size]]
                
                # stack the step data to form a window
                window_X = np.vstack(steps)
                
                # use the label from the last step in the window
                window_y = session_data[i+window_size-1][1]
                
                sequence_features = []
                # calculate statistical features for each sensor
                for j in range(window_X.shape[1]):
                    sensor_data = window_X[:, j]
                    
                    # calculate the power spectral density for frequency domain features
                    freqs, psd = signal.welch(sensor_data, fs=100)
                    
                    # extract time and frequency domain features
                    sequence_features.extend([
                        np.mean(sensor_data),          # mean
                        np.std(sensor_data),           # standard deviation
                        np.min(sensor_data),           # minimum value
                        np.max(sensor_data),           # maximum value
                        np.median(sensor_data),        # median
                        stats.skew(sensor_data),       # skewness (distribution asymmetry)
                        stats.kurtosis(sensor_data),   # kurtosis (distribution "tailedness")
                        np.percentile(sensor_data, 25), # 25th percentile
                        np.percentile(sensor_data, 75), # 75th percentile
                        np.ptp(sensor_data),           # peak-to-peak (max-min)
                        np.sum(psd),                   # total power
                        np.mean(psd),                  # mean power
                        np.max(psd),                   # max power
                        freqs[np.argmax(psd)]          # dominant frequency
                    ])
                
                # calculate window duration as the mean of metadata durations for the steps in the window
                window_duration = session_meta.iloc[i:i+window_size]['duration'].mean()
                
                # append the window duration feature directly to the window's features
                sequence_features.append(window_duration)
                
                # get the leg_side for this session
                leg_side = session_meta.iloc[i]['leg_side']
                
                # append the leg side to the features
                sequence_features.append(leg_side)
                
                # add extracted statistical features and labels to be output     
                features.append(sequence_features)
                labels.append(window_y.item())
        
        return np.array(features), np.array(labels)
    
    def __len__(self):
        """
        Return the number of windows in the dataset.
        
        Returns:
            - the number of windows
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a specific window and its label by index.
        
        Inputs:
            idx: index of the window to retrieve
            
        Returns:
            - a tensor of features and a tensor of labels
        """
        return (torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long))

def group_files(emg_files):
    """
    Group EMG files by collection ID and leg side.

    Input:
        - emg_files (list): List of file paths to EMG data files
        
    Returns:
        - a Dictionary with keys labeled "{collection_id}_{leg_side}" and values as lists of file paths
    """
    # the returned dictionary
    grouped_files = {}

    for file_path in emg_files:
        # extract just the filename without directory path
        file_name = os.path.basename(file_path)
        
        # extract the collection ID from filename
        match = re.search(r'(\d{2,4})', file_name)
        if match:
            # get the collection ID from the regex match
            collection_id = match.group(1)

            # determine the leg side based on the filename
            leg_side = "left" if "_left_" in file_name else "right"

            # create a unique identifier for this collection-leg combination
            collection_leg_id = f"{collection_id}_{leg_side}"
            
            # initialize the list for this collection-leg ID if it doesn't exist yet
            if collection_leg_id not in grouped_files:
                grouped_files[collection_leg_id] = []
            
            # add the file path to the appropriate group
            grouped_files[collection_leg_id].append(file_path)

    return grouped_files 

def get_sensor_labels(grouped_files):
    """
    Extract and collect all unique sensor labels from the grouped EMG files.

    Input:
        - grouped_files (dict): A dictionary of grouped files
                            
    Returns:
        - A set of unique sensor labels
   """
    # track all possible sensor labels
    all_sensor_labels = set()

    # collect all possible sensor labels
    for group, files in grouped_files.items():
        for file in files:
            # extract the sensor label from the filename using regex
            sensor_label = re.search(r'_emg_(\w+)\.csv', file).group(1)[1:]

            # exclude the vl sensor since it is missing from many collections
            if sensor_label != 'vl':
                # add the sensor label to the set
                all_sensor_labels.add(sensor_label)
    
    return all_sensor_labels

def process_and_combine_data(ms_folder, normative_folder):
    """
    Process and combine EMG data from MS patients and normative participants.

    This function:
    1. Collects all EMG files from both MS and normative folders
    2. Groups files by collection ID and leg side
    3. Processes each file to extract sensor data 
    4. Handles missing sensors by either padding or skipping sessions
    5. Combines all data into unified arrays with associated metadata

    Inputs:
        ms_folder (str): Path to the folder containing MS patient data
        normative_folder (str): Path to the folder containing normative participant data
        
    Returns:
        - sensor_data: Numpy array of shape (total_steps, 100, num_sensors)
        - metadata: Pandas Dataframe with metadata for each step
    """
    combined_data = []

    # collect all EMG files
    all_emg_files = glob.glob(os.path.join(ms_folder, "*_emg_*.csv")) + \
                    glob.glob(os.path.join(normative_folder, "*_emg_*.csv"))
    
    # group files by collection ID and leg side
    grouped_files = group_files(all_emg_files)

    # get all unique sensor labels across all files
    all_sensor_labels = get_sensor_labels(grouped_files)

    for group, files in grouped_files.items():
        dataframes = []
        sensors_present = set()

        for file in files:
            df = pd.read_csv(file)
            
            # extract the sensor label from filename
            sensor_label = re.search(r'_emg_(\w+)\.csv', file).group(1)[1:]

            # skip the 'vl' sensor data since it's missing from many collections
            if sensor_label == 'vl':
                continue

            # keep track of available sensors
            sensors_present.add(sensor_label)

            # keep the original 100 data points intact
            df = df.iloc[:, 2:102].values  # Shape (num_steps, 100)
            dataframes.append(df)

        # identify missing sensors
        missing_sensors = all_sensor_labels - sensors_present

        # drop rows for sessions missing 3 or more sensors (basically if more than half is missing)
        if len(missing_sensors) >= 3:
            print(f"Skipping session {group} (missing {len(missing_sensors)} sensors: {', '.join(missing_sensors)})")
            continue

        # add zero-padding for missing sensors (if fewer than 3 are missing)
        missing_cols = len(missing_sensors)
        if missing_cols > 0:
            print(f"Padding session {group} (missing {len(missing_sensors)} sensors: {', '.join(missing_sensors)})")
            zero_padding = np.zeros((dataframes[0].shape[0], 100, missing_cols))
            dataframes.append(zero_padding)

        # stack sensor data along axis 2 to maintain the shape (num_steps, 100, num_sensors)
        sensor_data = np.stack(dataframes, axis=-1)

        # extract the metadata for start, duration, etc.
        metadata = pd.read_csv(files[0])
        combined_df = pd.DataFrame({
            'start': metadata['start'],                    # start time of each step
            'duration': metadata['duration'],              # duration of each step
            'session_id': group,                           # session identifier
            'leg_side': 1 if '_left_' in files[0] else 0,  # left (1) or right (0) leg
            'has_ms': 1 if 'MSGaits' in files[0] else 0    # MS (1) or healthy (0)
        })

        combined_data.append((sensor_data, combined_df))

    # concatenate the data together to save the sensor data and metadata
    sensor_data_final = np.concatenate([entry[0] for entry in combined_data], axis=0)
    metadata_final = pd.concat([entry[1] for entry in combined_data]).reset_index(drop=True)

    return sensor_data_final, metadata_final

def get_train_test_datasets(X, y, metadata, window_size=4, test_size=0.1):
    '''
    This function creates the training and test datasets as well as performs feature extraction and train/test splits

    Inputs:
        - X: The feature set
        - y: The labels
        - window_size: The number of steps we want to analyze per inference
        - test_size: The size of the test set
    
    Returns:
        - train_dataset: The training dataset
        - test_dataset: The test dataset
        - scaler: The scaler used to perform normalization on the training set
    '''
    # split the data by session
    sessions = metadata['session_id'].values
    unique_sessions = np.unique(sessions)

    # create a mapping of session_id to MS status
    session_to_ms_status = {}
    
    for session_id in unique_sessions:
        # get all rows for this session
        session_mask = metadata['session_id'] == session_id
        
        # if any row has MS, the whole session is labeled as MS
        has_ms = any(metadata.loc[session_mask, 'has_ms'] == 1)
        session_to_ms_status[session_id] = 1 if has_ms else 0

    # create lists of session IDs by MS status
    ms_sessions = [s for s, status in session_to_ms_status.items() if status == 1]
    non_ms_sessions = [s for s, status in session_to_ms_status.items() if status == 0]

    # perform a stratified split on MS and non-MS sessions separately
    train_ms, test_ms = train_test_split(ms_sessions, test_size=test_size, random_state=42, shuffle=True)
    train_non_ms, test_non_ms = train_test_split(non_ms_sessions, test_size=test_size, random_state=42, shuffle=True)

    # combine MS and non-MS sessions for each split
    train_sessions = train_ms + train_non_ms
    test_sessions = test_ms + test_non_ms

    train_indices = metadata['session_id'].isin(train_sessions)
    test_indices = metadata['session_id'].isin(test_sessions)

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # session IDs for reference
    train_sessions_ids = sessions[train_indices]
    test_sessions_ids = sessions[test_indices]

    # create datasets
    train_dataset = SensorDataset(X_train, y_train, train_sessions_ids, metadata, window_size=window_size)
    test_dataset = SensorDataset(X_test, y_test, test_sessions_ids, metadata, window_size=window_size)

    # normalize the data
    scaler = StandardScaler()
    train_dataset.X = scaler.fit_transform(train_dataset.X)
    test_dataset.X = scaler.transform(test_dataset.X)

    return train_dataset, test_dataset, scaler

def prep_collection_for_inference(collection_path, scaler, window_size=4):
    """
    Prepare a single collection of EMG data to run inference on a trained model.

    This function:
    1. Loads and processes EMG files from a specific collection folder
    2. Validates data quality (e.g. missing sensors)
    3. Creates a SensorDataset with appropriate windowing
    4. Transforms features using the provided scaler

    Inputs:
        - collection_path (str): Path to a collection folder containing EMG files
        - scaler (StandardScaler): The scaler used to normalize the training data
        - window_size (int): Steps the model uses to analyze and predict MS
        
    Returns:
        - the scaled features ready for model inference
    """
    # find all EMG files in the collection folder
    emg_files = glob.glob(os.path.join(collection_path, "*_emg_*.csv"))

    # Group files by collection ID and leg side
    grouped = group_files(emg_files)

    # ensure there's exactly one session in the folder
    if len(grouped) != 1:
        raise ValueError("Expected only one session in the folder.")

    # extract the group ID and files
    group_id, files = list(grouped.items())[0]

    # get the sensor labels for this collection
    all_sensor_labels = get_sensor_labels({group_id: files})

    # process each file to extract sensor data
    dataframes = []
    sensors_seen = set()

    for file in files:
        # extract the sensor label from the filename
        sensor_label = re.search(r'_emg_(\w+)\.csv', file).group(1)[1:]

        # skip the 'vl' sensor data since we don't use it for training
        if sensor_label == 'vl':
            continue

        # track which sensors have been seen
        sensors_seen.add(sensor_label)

        # extract the 100 EMG data points for each step
        df = pd.read_csv(file).iloc[:, 2:102].values
        dataframes.append(df)

    # check for missing sensors
    missing = all_sensor_labels - sensors_seen

    # Raise an error if too many sensors are missing
    if len(missing) >= 3:
        raise ValueError(f"Too many missing sensors in session: {missing}")
    
    # Pad the data if less than 3 sensors are missing
    if len(missing) > 0:
        padding = np.zeros((dataframes[0].shape[0], 100, len(missing)))
        dataframes.append(padding)

    # stack sensor data along the third dimension
    sensor_data = np.stack(dataframes, axis=-1)

    # use the metadata from the first file
    meta_df = pd.read_csv(files[0])
    start = meta_df['start'].values
    duration = meta_df['duration'].values

    # extract the collection ID from the filename
    session_id_match = re.search(r'(\d{2,4})', files[0])

    # determine leg side
    leg_side = 1 if '_left_' in files[0] else 0

    # create the metadata df
    metadata = pd.DataFrame({
        'start': start,
        'duration': duration,
        'session_id': session_id_match.group(1) if session_id_match else 'unknown',
        'leg_side': leg_side,
        'has_ms': 0  # dummy prediction to match previous metadata
    })

    # create dummy labels to instantiate the dataset
    y_dummy = np.zeros(sensor_data.shape[0])

    # create the sensor dataset
    dataset = SensorDataset(sensor_data, y_dummy, metadata['session_id'].values, metadata, window_size)

    # normalize the feature set
    X_features = scaler.transform(dataset.X)

    return X_features

def get_data_root():
    """
    Determines the root data directory
    
    Returns:
        - the local data directory (one level up from this file, in a folder named 'data') if it exists. Otherwise, 
          it falls back to the data directory installed within the 'multiple_sclerosis_research' package.
    """
    # Construct the absolute path to the local data directory
    local_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Return local_data if it exists. Otherwise, use pkg_resources to locate the installed data directory
    return local_data if os.path.exists(local_data) else pkg_resources.resource_filename('multiple_sclerosis_research', 'data')

def run_data_preprocessing_pipeline(ms_folder='MSGaits', normative_folder='NormativeGaits', output_dir='processed_data'):
    '''
    Runs the entire data preprocessing pipeline and saves the processed sensor and metadata to their respective files

    Inputs:
        - ms_folder (str): The subfolder name within the data directory to find the MS data
        - normative_folder (str): The subfolder name within the data directory to find the normative data
        - output_dir (str): The subfolder name within the data directory to output the processed data
    '''
    # set the root directory
    root_dir = get_data_root()
    
    # set the input raw data directories
    ms_folder = os.path.join(root_dir, "raw_data", ms_folder)
    normative_folder = os.path.join(root_dir, "raw_data", normative_folder)

    # set the directory to store the processed data
    output_dir = os.path.join(root_dir, output_dir)

    # get the processed arrays to be stored
    sensor_data, metadata = process_and_combine_data(ms_folder, normative_folder)

    # save the processed data
    np.save(os.path.join(output_dir, "sensor_data.npy"), sensor_data)
    metadata.to_csv(os.path.join(output_dir, "combined_metadata.csv"), index=False)

    print(f"\nCombined dataset saved with {sensor_data.shape[0]} steps and {sensor_data.shape[2]} sensors.")

def main():
    # run the entire data preprocessing pipeline
    run_data_preprocessing_pipeline()

if __name__=='__main__':
    main()