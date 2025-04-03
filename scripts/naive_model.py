from data_preprocessing import get_train_test_datasets, prep_collection_for_inference
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def train_naive_model(train_dataset, test_dataset, save_path=''):
    '''
    Function to 'train' the naive model based on feature means.

    Inputs:
        - train_dataset: The training dataset to be used for computing mean features
        - test_dataset: The test dataset to predict on
        - save_path: The optional path for where to save the naive model
    '''
    # set the test labels
    y_test = test_dataset.y

    # compute the mean feature vector for MS and healthy samples in the training set
    mean_ms = np.mean(train_dataset.X[train_dataset.y == 1], axis=0)
    mean_healthy = np.mean(train_dataset.X[train_dataset.y == 0], axis=0)

    # compute the euclidean distance from each test sample to the MS and healthy means
    dist_to_ms = np.linalg.norm(test_dataset.X - mean_ms, axis=1)
    dist_to_healthy = np.linalg.norm(test_dataset.X - mean_healthy, axis=1)

    # predict MS if the sample is closer to the MS mean otherwise predict healthy
    y_pred = (dist_to_ms < dist_to_healthy).astype(int)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Naive Model Accuracy: {accuracy:.4f}")
    print(f"Naive Model Precision: {precision:.4f}")
    print(f"Naive Model Recall: {recall:.4f}")
    print(f"Naive Model F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # save the model
    if save_path != '':
        mean_model = {
            "mean_ms": mean_ms,
            "mean_healthy": mean_healthy
        }

        joblib.dump(mean_model, save_path)

def predict_naive_model(collection_path, model, scaler, window_size=4):
    '''
    Function to run inference on the naive model

    Inputs:
        - collection_path: the path for the collection of csvs to run inference on
        - model: The naive model to use for inference
        - scaler: The scaler used to scale the training data
        - window_size: The rolling window used for prediction (i.e. the steps to look at at a time)
    
    Returns:
        - The naive model prediction (1 for MS, 0 for healthy)
    '''
    # get the mean feature values for ms patients
    ms_mean = model["mean_ms"]

    # get the mean feature values for healthy patients
    healthy_mean = model["mean_healthy"]

    # extract the statistical features from the collection to match the training data
    X_features = prep_collection_for_inference(collection_path, scaler, window_size)

    # calculate the distance of the collection features to the MS mean and healthy means
    dist_to_ms = np.linalg.norm(X_features - ms_mean, axis=1)
    dist_to_healthy = np.linalg.norm(X_features - healthy_mean, axis=1)

    # get the prediction from the naive model
    preds = (dist_to_ms < dist_to_healthy).astype(int)

    return int(np.round(np.mean(preds)))

def main():
    # set the root and data directories
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(root_dir, "data", "processed_data")

    # load the sensor data, metadata and get the ground-truth labels
    X = np.load(os.path.join(data_dir, "sensor_data.npy"))
    metadata = pd.read_csv(os.path.join(data_dir, "combined_metadata.csv"))
    y = metadata['has_ms'].values

    # extract the statistical features from the data for training
    train_dataset, test_dataset, _ = get_train_test_datasets(X, y, metadata)

    # 'train' the naive model
    train_naive_model(train_dataset, test_dataset)

if __name__=='__main__':
    main()