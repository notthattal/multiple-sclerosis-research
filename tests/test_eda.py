import pytest
import os
import pandas as pd
from eda import is_left

def get_subject_number(text):
    '''
    Gets the subject number from the name of the folder

    Inputs:
        - text (str): The folder name for which to extract the number
    
    Returns:
        - The integer value of the subject corresponding to that subject's alias ID
    '''
    # string corresponding to the integer value from the folder name
    number = ''

    # go through the reversed string (since all subject folders are named 'Subject{number}')
    for char in reversed(text):
        # Get the digit, assign it to the string and break if we hit a non-numeric digit
        if char.isdigit():
            number = char + number
        else:
            break
    
    # convert the string to an int and return it
    return int(number)

def test_is_left():
    '''
    Test the "is_left" function in eda.py. This verifies through double checking the SubjectInformation.csv that the is_left function
    properly returns True or False
    '''
    # assign the folder which contains subject CSVs and the get the dataframe with subject information
    data_folder = os.path.join(os.path.dirname(__file__), '../data')
    subject_df = pd.read_csv(os.path.join(data_folder, 'SubjectInformation.csv'))

    # go through the data folder
    for item in os.listdir(data_folder):
        # skip the SubjectInformation.csv
        if not os.path.isdir(os.path.join(data_folder, item)):
            continue

        # grab a folder to check if the subject has a left or right leg impairment
        subject_folder = os.path.join(data_folder, item, 'FinalAssessment/T25FW')

        # check if the subject has a left or right leg impairment in the CSV
        impaired_limb_val = subject_df.loc[subject_df['ID'] == get_subject_number(item), 'Impaired_Limb'].values[0]
        check_left = impaired_limb_val == 0

        # verify that both the impairment found through the dataframe and the subject_folder are the same
        assert check_left == is_left(subject_folder)