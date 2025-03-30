from data_preprocessing import get_train_test_datasets
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def train_naive_model(train_dataset, test_dataset):
    '''
    Function to 'train' the mean model

    Inputs:
        - train_dataset: The training dataset to be used as the basis for the mean model to make predictions from
        - test_dataset: The test dataset to predict on
    '''
    # the ground-truth labels for the training and test sets
    y_train = train_dataset.y
    y_test = test_dataset.y

    # Calculate the mean of the training data and use it to predict on the test set (mean(training data) >= 0.5 means MS otherwise health)
    mean_prediction = int(np.mean(y_train) >= 0.5)

    # get the predictions on the test set
    y_pred = np.full_like(y_test, fill_value=mean_prediction)

    # calculate the evaluation metrics for the mean model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # print the evaluation metrics and classification scores for the mean model
    print(f"Mean Model Accuracy: {accuracy:.4f}")
    print(f"Mean Model Precision: {precision:.4f}")
    print(f"Mean Model Recall: {recall:.4f}")
    print(f"Mean Model F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

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