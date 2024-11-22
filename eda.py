import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def is_left(folder_path):
    '''
    Checks if a user has a left leg or right leg impairment

    Inputs:
        - folder_path (str): The path for which to check
    '''
    # Iterate through all files in the folder
    for file in os.listdir(folder_path):
        # Check if "emg" is in the name and it's a CSV (only EMG files have specifications for which leg is being recorded)
        if "emg" in file.lower() and file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                
                # Check if it is left-sided or right sided
                if 'lvl' in df.columns:
                    return True
                else:
                    return False
            except Exception as e:
                print(f"Could not read file '{file}'. Error: {e}")

def plot_emg_for_assessment(parent_dir, protocol, assessment, column):
    '''
    Plots the stimulated walk data for a specific EMG and a specific assessment

    Inputs:
        - parent_dir (str): The data directory
        - protocol (str): If it's for the 6 minute or timed 25-foot walk
        - assessment (str): Which assessment to plot (InitialAssessment, MidpointAssessment or FinalAssessment)
        - column (str): Which EMG to plot (e.g. 'lhl' = left hamstring lateral)
    '''
    assisted_dfs = {}
    for subject_folder in os.listdir(parent_dir):
        # the directory for a specific subject
        subject_path = Path(os.path.join(parent_dir, subject_folder))

        if not os.path.isdir(subject_path):
            continue

        # the proper subdirectory
        walk_test_folder = Path(os.path.join(parent_dir, subject_folder, assessment)) / protocol

        # skip missing assessments (Cionic issue. A discrepancy between the data they have uploaded and their reported findings)
        # Did not remove the rows because they are working on getting it fixed
        if not os.path.isdir(walk_test_folder):
            continue

        for file in os.listdir(walk_test_folder):
            # only want the EMG data
            if 'imu' in file:
                continue

            file_path = walk_test_folder / file
            df = pd.read_csv(file_path)

            if column not in df.columns:
                continue
            
            # EMGs were only placed on the shank and looking at the stimulated walks
            if 'shank' in file and ('_assisted' in file or '_stimwalk' in file):
                assisted_dfs[subject_path.name] = df.copy()
    
    # crop x-axis to the minimum length for easier viewing
    min_length = float('inf')
    for assisted_df in assisted_dfs.values():
        min_length = min(min_length, assisted_df.shape[0])

    # create the plot
    for label, assisted_df in assisted_dfs.items():
        assisted_df = assisted_df.copy() 
        assisted_df = assisted_df.iloc[:min_length]
        assisted_df['scaled_elapsed_s'] = (assisted_df['elapsed_s'] - assisted_df['elapsed_s'].min()) / (assisted_df['elapsed_s'].max() - assisted_df['elapsed_s'].min())
        assisted_df['Smoothed'] = assisted_df.loc[:, column].rolling(window=min_length // 5, center=True).mean()
        assisted_df['Std'] = assisted_df.loc[:, column].rolling(window=min_length // 5, center=True).std()
        plt.plot(assisted_df['scaled_elapsed_s'], assisted_df['Smoothed'], label=label)
        plt.fill_between(
            assisted_df['scaled_elapsed_s'],
            assisted_df['Smoothed'] - assisted_df['Std'],
            assisted_df['Smoothed'] + assisted_df['Std'],
            alpha=0.2)
        
    plt.title(f'{assessment} {column} - {protocol}')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_emg_assessment_for_all_cols(parent_dir, protocol, assessment):
    '''
    Plots the stimulated walk data for all EMGs for a specific assessment

    Inputs:
        - parent_dir (str): The data directory
        - protocol (str): If it's for the 6 minute or timed 25-foot walk
        - assessment (str): Which assessment to plot (InitialAssessment, MidpointAssessment or FinalAssessment)    
    '''

    assisted_dfs_l = {}
    assisted_dfs_r = {}
    columns_l = []
    columns_r = []
    for subject_folder in os.listdir(parent_dir):
        # the directory for a specific subject
        subject_path = Path(os.path.join(parent_dir, subject_folder))

        if not os.path.isdir(subject_path):
            continue

        # the proper subdirectory
        walk_test_folder = Path(os.path.join(parent_dir, subject_folder, assessment)) / protocol

        # skip missing assessments (Cionic issue. A discrepancy between the data they have uploaded and their reported findings)
        # Did not remove the rows because they are working on getting it fixed
        if not os.path.isdir(walk_test_folder):
            continue

        # check if the subject has a left-leg or right-leg impairment
        left = is_left(str(walk_test_folder))

        for file in os.listdir(walk_test_folder):
            # only want the EMG data
            if 'imu' in file:
                continue

            file_path = walk_test_folder / file
            df = pd.read_csv(file_path)
            if left and len(columns_l) == 0:
                columns_l = df.columns
            elif not left and len(columns_r) == 0:
                columns_r = df.columns
            
            # EMGs were only placed on the shank and looking at the stimulated walks
            if 'shank' in file and ('_assisted' in file or '_stimwalk' in file):
                if left:
                    assisted_dfs_l[subject_path.name] = df.copy()
                else:
                    assisted_dfs_r[subject_path.name] = df.copy()
            
    for i in range(2):
        assisted_dfs = assisted_dfs_l if i == 0 else assisted_dfs_r
        columns = columns_l if i == 0 else columns_r
        for column in columns:
            if column == 'elapsed_s':
                continue

            # crop x-axis to the minimum length for easier viewing
            min_length = float('inf')
            for assisted_df in assisted_dfs.values():
                min_length = min(min_length, assisted_df.shape[0])

            # create the plot
            for label, assisted_df in assisted_dfs.items():
                assisted_df = assisted_df.copy() 
                assisted_df = assisted_df.iloc[:min_length]
                assisted_df['scaled_elapsed_s'] = (assisted_df['elapsed_s'] - assisted_df['elapsed_s'].min()) / (assisted_df['elapsed_s'].max() - assisted_df['elapsed_s'].min())
                assisted_df['Smoothed'] = assisted_df.loc[:, column].rolling(window=min_length // 5, center=True).mean()
                assisted_df['Std'] = assisted_df.loc[:, column].rolling(window=min_length // 5, center=True).std()
                plt.plot(assisted_df['scaled_elapsed_s'], assisted_df['Smoothed'], label=label)
                plt.fill_between(
                    assisted_df['scaled_elapsed_s'],
                    assisted_df['Smoothed'] - assisted_df['Std'],
                    assisted_df['Smoothed'] + assisted_df['Std'],
                    alpha=0.2)
                
            plt.title(f'{assessment} {column} - {protocol}')
            plt.xlabel('Time')
            plt.legend()
            plt.tight_layout()
            plt.show()

def plot_emg(parent_dir, protocol, column):
    '''
    Plot the EMG data for one subject and one EMG over all of that subject's assessments

    Inputs:
        - parent_dir (str): The data directory
        - protocol (str): If it's for the 6 minute or timed 25-foot walk
        - column (str): Which EMG to plot (e.g. 'lhl' = left hamstring lateral)
    '''
    for subject_folder in os.listdir(parent_dir):
        # the directory for a specific subject
        subject_path = Path(os.path.join(parent_dir, subject_folder))

        if not os.path.isdir(subject_path):
            continue

        assisted_dfs = {}
        for assessment_folder in subject_path.iterdir():
            # the proper subdirectory
            walk_test_folder = assessment_folder / protocol

            # skip missing assessments (Cionic issue. A discrepancy between the data they have uploaded and their reported findings)
            # Did not remove the rows because they are working on getting it fixed
            if not os.path.isdir(walk_test_folder):
                continue

            for file in os.listdir(walk_test_folder):
                # only want the EMG data
                if 'imu' in file:
                    continue

                file_path = walk_test_folder / file
                df = pd.read_csv(file_path)
                
                # EMGs were only placed on the shank and looking at the stimulated walks
                if 'shank' in file and ('_assisted' in file or '_stimwalk' in file):
                    assisted_dfs[assessment_folder.name] = df.copy()
        
        should_skip = False
        min_length = float('inf')
        for assisted_df in assisted_dfs.values():
            if column not in assisted_df.columns:
                should_skip = True
                break
            
            # crop x-axis to the minimum length for easier viewing
            min_length = min(min_length, assisted_df.shape[0])

        # skip plotting unnecessary columns
        if should_skip:
            continue

        # create the plot
        for label, assisted_df in assisted_dfs.items():
            assisted_df = assisted_df.copy() 
            assisted_df = assisted_df.iloc[:min_length]
            assisted_df['scaled_elapsed_s'] = (assisted_df['elapsed_s'] - assisted_df['elapsed_s'].min()) / (assisted_df['elapsed_s'].max() - assisted_df['elapsed_s'].min())
            assisted_df['Smoothed'] = assisted_df.loc[:, column].rolling(window=min_length // 5, center=True).mean()
            plt.plot(assisted_df['scaled_elapsed_s'], assisted_df['Smoothed'], label=label)
            
        plt.title(f'{subject_folder} {column} - {protocol}')
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_mean_emg(parent_dir, protocol):
    '''
    Plot the mean EMG data for all subjects over all of that subject's assessments

    Inputs:
        - parent_dir (str): The data directory
        - protocol (str): If it's for the 6 minute or timed 25-foot walk
    '''
    unassisted_means_l = pd.DataFrame()
    unassisted_means_r = pd.DataFrame()
    assisted_means_l = pd.DataFrame()
    assisted_means_r = pd.DataFrame()
    columns_l = []
    columns_r = []

    for subject_folder in os.listdir(parent_dir):
        # the directory for a specific subject
        subject_path = Path(os.path.join(parent_dir, subject_folder))

        if not os.path.isdir(subject_path):
            continue

        for assessment_folder in subject_path.iterdir():
            # the proper subdirectory
            walk_test_folder = assessment_folder / protocol

            # skip missing assessments (Cionic issue. A discrepancy between the data they have uploaded and their reported findings)
            # Did not remove the rows because they are working on getting it fixed
            if not os.path.isdir(walk_test_folder):
                continue

            # check if the subject has a left-leg or right-leg impairment
            left = is_left(str(walk_test_folder))

            for file in os.listdir(walk_test_folder):
                # only want the EMG data
                if 'imu' in file:
                    continue

                file_path = walk_test_folder / file
                df = pd.read_csv(file_path)
                if left:
                    columns_l = df.columns
                else:
                    columns_r = df.columns

                temp_df = df.mean().to_frame().T
                temp_df['assessment'] = assessment_folder.name
                
                # EMGs were only placed on the shank and looking at the stimulated walks
                if 'shank' in file:
                    if '_unassisted' in file or 'unstim' in file:
                        if left:
                            unassisted_means_l = pd.concat([unassisted_means_l, temp_df], ignore_index=True)
                        else:
                            unassisted_means_r = pd.concat([unassisted_means_r, temp_df], ignore_index=True)
                    elif '_assisted' in file or '_stimwalk' in file:
                        if left:
                            assisted_means_l = pd.concat([assisted_means_l, temp_df], ignore_index=True)
                        else:
                            assisted_means_r = pd.concat([assisted_means_r, temp_df], ignore_index=True)

    unassisted_means_l = unassisted_means_l.groupby('assessment', as_index=False).mean()
    unassisted_means_r = unassisted_means_r.groupby('assessment', as_index=False).mean()
    assisted_means_l = assisted_means_l.groupby('assessment', as_index=False).mean()
    assisted_means_r = assisted_means_r.groupby('assessment', as_index=False).mean()

    assessments = ['InitialAssessment', 'MidpointAssessment', 'FinalAssessment']

    # plot the left and right assessments
    for i in range(2):
        unassisted_means = unassisted_means_l if i == 0 else unassisted_means_r
        assisted_means = assisted_means_l if i == 0 else assisted_means_r
        columns = columns_l if i == 0 else columns_r

        unassisted_means['assessment'] = pd.Categorical(
            unassisted_means['assessment'], 
            categories=assessments, 
            ordered=True
        )

        assisted_means['assessment'] = pd.Categorical(
            assisted_means['assessment'], 
            categories=assessments, 
            ordered=True
        )

        # sort by assessment to show the change from Initial to Final
        unassisted_means = unassisted_means.sort_values(by='assessment').reset_index(drop=True)
        assisted_means = assisted_means.sort_values(by='assessment').reset_index(drop=True)

        # create the plot
        for column in columns:
            if column != "elapsed_s":
                plt.figure(figsize=(8, 5))
                
                # Extract means for the current column across all files
                unassisted_means[column].plot(kind='line', label='Unassisted', marker='o')
                assisted_means[column].plot(kind='line', label='Assisted', marker='o')
                
                plt.title(f'Mean {column} - Assisted vs Unassisted {protocol}')
                plt.xlabel('Assessment')
                plt.ylabel(f'Mean {column}')
                plt.xticks(ticks=range(len(unassisted_means)), labels=unassisted_means['assessment'], rotation=0)
                plt.legend()
                plt.tight_layout()
                plt.show()


def main():
    data_dir = './data'
    protocol = 'T25FW'
    column = 'lpl'
    assessment = 'FinalAssessment'
    plot_emg(data_dir, protocol, column)

if __name__ == '__main__':
    main()