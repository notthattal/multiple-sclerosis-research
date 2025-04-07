import sys
import os

# this line is added to be able to call relative imports from the streamlit website
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts"))
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

import zipfile
import tempfile
import joblib
from scripts.traditional_model import predict_traditional_model
from scripts.naive_model import predict_naive_model
from scripts.deep_learning_model import predict_dl_model
import shutil
import streamlit as st

def load_model(model_type):
    '''
    Load the appropriate model based on a user selection

    Input:
        - model_type: The string to tell which model to select
    
    Returns:
        - the model, scaler used for training and the inference function
    '''
    # get the saved scaler used for training
    scaler = joblib.load("models/scaler.pkl")
    
    # return the selected model, scaler and the inference function to be called
    if model_type == "Naive Model":
        print('test1')
        model = joblib.load("models/ms_naive_model.pkl")
        return model, scaler, predict_naive_model
    elif model_type == "Traditional Model":
        print('test2')
        model = joblib.load("models/ms_traditional_model.pkl")
        return model, scaler, predict_traditional_model
    elif model_type == "Deep Learning Model":
        print('test3')
        model_path = "models/ms_dl_model.pth"
        return model_path, scaler, predict_dl_model
    else:
        st.warning("Model type not implemented.")
        return None, None, None


def get_prediction(collection_path, model_type):
    '''
    Run inference for the given data using the selected model

    Inputs:
        - collection_path: the path to the loaded collection zip
        - model_type: The string to tell which model to select
    
    Returns:
        - The selected model's prediction as a probability
    '''
    try:
        # load the model, scaler and inference function
        model, scaler, inference_func = load_model(model_type)
        
        if model is None:
            return None
            
        # get the predicted probabilities
        probs = inference_func(collection_path, model, scaler)

        return probs
        
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None


def display_prediction(probs):
    ''' 
    Displays the prediction results to the user

    Inputs:
        - probs: The predicted probability of having MS
    '''
    if probs is not None:
        # convert prediction to integer
        prediction = int(probs >= 0.5)

        # display prediction
        if prediction == 1:
            st.error(f"Prediction: MS Detected with probability ({probs:0.2f})")
        else:
            st.success(f"Prediction: No MS Detected with probability ({(1 - probs):0.2f})")


def extract_zip(uploaded_zip):
    '''
    Extract the uploaded zip file to a temporary directory

    Inputs:
        - uploaded_zip: The uploaded zip file containing all collection CSVs
    
    Returns:
        - the collection path
    '''
    # clean up previous temporary directory if it exists
    if 'temp_dir' in st.session_state and st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir)
    
    # create a new temporary directory
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = temp_dir
    
    # extract files from zip
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(temp_dir)

    st.success("ZIP extracted successfully.")

    # get the extracted directory (assumes the zip contains one folder)
    extracted_items = os.listdir(temp_dir)
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
        collection_path = os.path.join(temp_dir, extracted_items[0])
    else:
        # create the directory structure expected by the model
        session_dir = os.path.join(temp_dir, "session")
        os.makedirs(session_dir, exist_ok=True)
        
        # move all files to this directory
        for item in extracted_items:
            src = os.path.join(temp_dir, item)
            dst = os.path.join(session_dir, item)
            if os.path.isfile(src):
                shutil.move(src, dst)
        
        collection_path = session_dir
    
    return collection_path

def setup_sidebar():
    '''
    Sets up the sidebar as a model selector
    '''
    # set sidebar title
    st.sidebar.title("Model Options")

    # set model options
    return st.sidebar.selectbox(
        "Select Model Type",
        ("Naive Model", "Traditional Model", "Deep Learning Model")
    )

def main():
    # set up the sidebar
    model_type = setup_sidebar()
    
    # set the title screen
    st.title("Multiple Sclerosis Classification")
    
    # create the section to upload zip files
    uploaded_zip = st.file_uploader("Upload a collection", type="zip")
    
    if uploaded_zip is not None:
        # extract files from the uploaded zip
        collection_path = extract_zip(uploaded_zip)
        
        # run inference on the uploaded collection
        if st.button("Get Prediction"):
            probs = get_prediction(collection_path, model_type)
            display_prediction(probs)

if __name__ == "__main__":
    main()