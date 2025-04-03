from data_preprocessing import get_train_test_datasets, prep_collection_for_inference, set_seeds
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

'''
An MLP model to be used to classify multiple sclerosis on statistical features extracted from Cionic's collections

Class Members:
    - hidden_layers (nn.Sequential): The hidden layers of the model
    - output_layer (nn.Linear): The final linear layer of the model
 '''
class MSDeepLearningClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout=0.5, num_classes=2):
        super(MSDeepLearningClassifier, self).__init__()

        # build the hidden layers
        self.hidden_layers = self.build_hidden_layers(input_size, hidden_sizes, dropout)

        # instantiate the final linear layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
    
    def build_hidden_layers(self, input_size, hidden_sizes, dropout):
        '''
        Builds the hidden layers for the MLP Model

        Inputs:
            - input_size: the initial input size to the model
            - hidden_sizes: The subsequent layers' number of neurons
            - dropout: The dropout percentage used for regularization
        
        Returns:
            - The hidden layers for the model
        '''
        # the array that will hold the hidden layer data
        layers = []
        
        # the input size for the layer
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # create the linear layer for the model with the input size being the output size of the previous layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # add batch norm to the hidden layer
            layers.append(nn.BatchNorm1d(hidden_size))

            # add the activation function to be used
            layers.append(nn.ReLU())

            # add dropout
            layers.append(nn.Dropout(dropout))

            # set the new input to be this layer's output size
            prev_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        '''
        Forward pass for the model

        Inputs:
            x: The input to the model
        
        Returns:
            x: The output of the forward pass
        '''

        # flatten the input to 1D
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # pass through hidden layers
        x = self.hidden_layers(x)
        
        # get the outputs from the final layer
        x = self.output_layer(x)
        
        # return predictions
        return x
    
def train(model, train_loader, criterion, optimizer, device, epochs=50, scheduler=None):
    '''
    The training loop for the deep learning model

    Inputs:
        - model: The chosen untrained deep learning model
        - train_loader: The data loader that contains the training data
        - criterion: The criterion for which to calculate the loss
        - optimizer: The chosen optimizer to be used
        - device: The device to run training on
        - epochs: The number of epochs to train for
        - scheduler: The learning rate scheduler to be used for training
    
    Returns:
        - model: The trained deep learning model
    '''
    # set model to training mode
    model.train()

    for epoch in range(epochs):
        # instantiate training loss for this epoch
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # move batch data to the specified device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # clear previous gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(X_batch)

            # compute loss
            loss = criterion(outputs, y_batch)

            # backward pass
            loss.backward()

            # update model weights
            optimizer.step()
            
            # track accumulated loss
            train_loss += loss.item()
        
        # get the average loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        
        # print metrics for this epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}')
        
        # update the learning rate if a scheduler is provided
        if scheduler is not None:
            scheduler.step(avg_train_loss)
    
    return model

def predict_dl_model(collection_path, model_path, scaler, device=torch.device("cpu"), window_size=4):
    '''
    Run inference for a given collection on the deep learning model

    Inputs:
        - collection_path: The path to the collection CSVs to generate a prediction from
        - model: The trained deep learning model
        - scaler: The scaler used to normalize the training data
        - window_size: The number of steps being analyzed at a given time
    
    Returns:
        - The majority vote prediction over all windows (1 = MS, 0 = Healthy) 
    '''
    # extract the statistical features for the collection to match the input the model was trained on
    X_features = prep_collection_for_inference(collection_path, scaler, window_size)

    # get the input size to the model
    input_size = X_features.shape[1]

    # instantiate the model
    model = MSDeepLearningClassifier(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # set model to evaluation mode
    model.eval()

    # disable gradient tracking for inference
    with torch.no_grad():
        # convert the inputs to a tensor and move it to the specified device
        inputs = torch.tensor(X_features, dtype=torch.float32).to(device)

        # forward pass for the model
        outputs = model(inputs)

        # get the predicted class labels
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # return the majority prediction as a single binary label
    return int(np.round(np.mean(preds)))

def evaluate(model, test_loader, device):
    '''
    Function to run evaluation on a test set for the deep learning model 

    Inputs:
        - model: The trained deep learning model
        - test_loader: The data loader for the test set
        - device: The device to be used
    '''
    # set model to evaluation mode
    model.eval()

    # arrays to hold the predictions and ground-truth labels
    all_preds = []
    all_labels = []
    
    # disable gradient tracking for evaluation
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # move the data to the specified device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # get the predictions from the model
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            
            # add the predictions and ground-truth labels to their respective arrays
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # get the evaluation metrics for the test set
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # print the evaluation metrics and classification report
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

def train_dl_model(train_dataset, test_dataset, device, save_model_path='', batch_size=32):
    '''
    Function to setup and train the deep learning model given a training and test set

    Inputs:
        - train_dataset: The training dataset
        - test_dataset: The test dataset
        - save_model_path: The optional path to save the model
        - batch_size: The batch size to setup the DataLoaders with
        - device: The device to run training on
    
    Returns:
        - model: the trained deep learning model
    '''
    # setup the training and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # instantiate the MLP model and move it to the appropriate device
    input_size = train_dataset[0][0].shape[0]
    model = MSDeepLearningClassifier(input_size).to(device)

    # set the loss criterion
    criterion = nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # define the lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # train the model
    print("Starting training...")
    model = train(model, train_loader, criterion, optimizer, device, epochs=50, scheduler=scheduler)

    # run evaluation on the model
    evaluate(model, test_loader, device)

    # save the model if given a specified output path
    if save_model_path != '':
        torch.save(model.state_dict(), save_model_path)
        print(f"Deep learning model saved to {save_model_path}")

    return model

def main():
    #set the random seeds
    set_seeds()

    # set the device to mps if available otherwise cpu
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # set the root and data directories
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(root_dir, "data", "processed_data")

    # load the sensor data, metadata and get the ground-truth labels
    X = np.load(os.path.join(data_dir, "sensor_data.npy"))
    metadata = pd.read_csv(os.path.join(data_dir, "combined_metadata.csv"))
    y = metadata['has_ms'].values

    # extract the statistical features from the data for training
    train_dataset, test_dataset, _ = get_train_test_datasets(X, y, metadata)

    # train the deep learning model
    train_dl_model(train_dataset, test_dataset, device)

if __name__=='__main__':
    main()