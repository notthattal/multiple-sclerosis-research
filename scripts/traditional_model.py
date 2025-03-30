from data_preprocessing import get_train_test_datasets, prep_collection_for_inference, set_seeds
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def train_traditional_model(train_dataset, test_dataset=None, save_path=''):
    '''
    Function to instantiate and train the traditional MS predictive model

    Inputs:
        - train_dataset (SensorDataset): The training data for the model
        - test_dataset (SensorDataset): The optional test dataset to run evaluation on
        - save_path (str): The optional string for where to save the model after training
    
    Returns:
        - ensemble_gb: The trained ensemble model of GradientBoostingClassifiers
    '''
    # retrieve the training features and labels from the dataset
    X_train, y_train = train_dataset.X, train_dataset.y

    # instantiate the base GradientBoostingClassifier for which to ensemble
    base_gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.2,
        max_depth=3,
        min_samples_split=2,
        random_state=42
    )

    # Create an ensemble of multiple instances of the base_gb model each trained on 90% of the data using with bootstrap sampling
    ensemble_gb = BaggingClassifier(
        estimator=base_gb,
        n_estimators=5,
        max_samples=0.9,
        bootstrap=True,
        random_state=42
    )

    # train the ensemble on the training data
    ensemble_gb.fit(X_train, y_train)

    # if a test dataset was passed into the function, output the evaluation metrics of the model on the test set
    if test_dataset:
        evaluate_traditional_model(ensemble_gb, test_dataset)

    # if a save path was passed into the function, save the model to the specified path
    if save_path != '':
        save_model(ensemble_gb, save_path)

    return ensemble_gb

def evaluate_traditional_model(model, test_dataset):
    '''
    Function to print out evaluation metrics for a trained traditional model on the test dataset
    
    Inputs:
        - model (BaggingClassifier): The trained ensemble of models
        - test_dataset (SensorDataset): The dataset to run evaluation on
    '''
    # get the test features and labels
    X_test, y_test = test_dataset.X, test_dataset.y

    # get the model predictions on the test data
    y_test_pred = model.predict(X_test)

    # get the accuracy, precision, recall and F1 for the test data in comparison to the ground truth labels
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # print out the metrics found above and the classification report
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

def save_model(model, output_path):
    '''
    Function to save the traditional model

    Inputs:
        - model (BaggingClassifier): The trained ensemble tree model
        - output_path (str): The path to output the saved file
    '''
    # open a file at output_path in binary write mode
    with open(output_path, 'wb') as file:
        # save the model
        pickle.dump(model, file)

    print(f"Model saved to {output_path}")

def predict_traditional_model(collection_path, model, scaler, window_size=4):
    '''
    Function to run inference on the traditional model
    
    Inputs:
        - collection_path (str): The path where the collection CSVs are stored
        - model (BaggingClassifier): The trained ensemble model
        - scaler (StandardScaler): The scaler that was used to normalize the training data
        - window_size (int): The window of walking steps used in the training data
    
    Returns:
        - prediction (int): Whether the model predicted MS or not (1 = MS, 0 = Healthy)
    '''
    # extract the statistical features from the collection to match the training data
    X_features = prep_collection_for_inference(collection_path, scaler, window_size)

    # get predictions for each window of the training data
    preds = model.predict(X_features)

    # predict if MS or by getting a majority vote over all the windows predicted for this collection
    prediction = int(np.round(np.mean(preds)))

    return prediction

def main():
    #set the random seeds
    set_seeds()

    # establish the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # set the data directory
    data_dir = os.path.join(root_dir, "data", "processed_data")

    # load the sensor data, metadata and labels
    X = np.load(os.path.join(data_dir, "sensor_data.npy"))
    metadata = pd.read_csv(os.path.join(data_dir, "combined_metadata.csv"))
    y = metadata['has_ms'].values

    # extract statistical features from the loaded data and split to train and test sets
    train_dataset, test_dataset, _ = get_train_test_datasets(X, y, metadata)

    # train the traditional model using the test dataset for evaluation purposes
    train_traditional_model(train_dataset, test_dataset)

if __name__=='__main__':
    main()