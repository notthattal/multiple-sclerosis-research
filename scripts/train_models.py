from data_preprocessing import get_train_test_datasets, set_seeds, run_data_preprocessing_pipeline
from deep_learning_model import train_dl_model
from naive_model import train_naive_model
import numpy as np
import os
import pandas as pd
import pickle
import torch
from traditional_model import train_traditional_model

def main():
    # set the seeds
    set_seeds()

    # set the device to mps if available otherwise cpu
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # run the data preprocessing pipeline
    run_data_preprocessing_pipeline()

    # set the root, data and models directories
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, "data", "processed_data")
    models_dir = os.path.join(root_dir, "models")

    # create the models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # set the model save paths
    naive_model_save_path = os.path.join(models_dir, 'ms_naive_model.pkl')
    traditional_model_save_path = os.path.join(models_dir, 'ms_traditional_model.pkl')
    dl_model_save_path = os.path.join(models_dir, 'ms_dl_model.pth')

    # load the sensor data, metadata and get the ground-truth labels
    X = np.load(os.path.join(data_dir, "sensor_data.npy"))
    metadata = pd.read_csv(os.path.join(data_dir, "combined_metadata.csv"))
    y = metadata['has_ms'].values

    # extract the statistical features from the data for training
    train_dataset, test_dataset, scaler = get_train_test_datasets(X, y, metadata)

    # train the naive model
    train_naive_model(train_dataset, test_dataset, naive_model_save_path)
    
    # train and save the traditional model
    train_traditional_model(train_dataset, test_dataset, traditional_model_save_path)
    
    # train and save the deep learning model
    train_dl_model(train_dataset, test_dataset, device, dl_model_save_path)

    # save the scaler that was used to normalize the training data
    scaler_save_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_save_path}")

if __name__=='__main__':
    main()