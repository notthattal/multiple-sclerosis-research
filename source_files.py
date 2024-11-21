from scripts import download as dl
import os
from pathlib import Path
import shutil
import pandas as pd

def parse_data(data_dir, out_dir, subjects_df):
    '''
    This function moves all the files from the download directory into the folder structure used to upload the CSVs
    to the site in which it's being open sourced

    Inputs:
        - data_dir (str): The path to the source data folder
        - out_dir (str): The path to the directory the files should be moved to
        - subjects_df (pd.Dataframe): The dataframe that has the subject and each test and collection associated to that subject
    '''
    # the path to the actual data
    study_path = Path(data_dir) / 'cionic/MSPilotCSU'
    for i in range(subjects_df.shape[0]):
        # get the name of the subject folder
        subject = 'Subject' + str(subjects_df.loc[i, 'ID'])
        
        # get the name of the test taken for this collection
        proto = subjects_df.loc[i, 'Protocol'].split(' ')
        
        # get the collection number
        collection = str(subjects_df.loc[i, 'Collection'])
        
        # set the correct source and destination paths associated with this row
        destination_path = os.path.join(out_dir, subject, proto[1] + 'Assessment', proto[0])
        source_path = os.path.join(study_path, proto[0], collection)

        # handle missing data issues with Cionic
        if not os.path.isdir(source_path):
            continue

        # make the directory if it doesn't exist
        os.makedirs(destination_path, exist_ok=True) 

        # move the csv files to the destination directory
        for file_name in os.listdir(source_path):
            if not file_name.endswith(".csv"):
                continue

            shutil.move(os.path.join(source_path, file_name), os.path.join(destination_path, file_name))

        print(f"File moved from '{source_path}' to '{destination_path}'.")

def name_file(folder_path):
    '''
    This modifies the file names to a shortened version that has a more meaningful name

    Inputs:
        - folder_path (Path): The path to the folder for which to rename files
    '''
    # check if this folder is for an impairment in the right or left leg
    left = is_left(folder_path)

    print('renaming files in: ' + str(folder_path))
    for file in os.listdir(folder_path):
        # only rename CSVs
        if not file.endswith(".csv"):
            continue

        # get the file number for naming puposes
        file_num = file.split('_')[3]
        # add to the new file name if it is Left or Right
        new_file_name = "l_" if left else "r_"

        # specify which part of the lef the sensor was placed
        if file.startswith('DC'):
            new_file_name += 'thigh_'
        elif file.startswith('SI'):
            new_file_name += 'shank_'
        
        # list if it was an IMU or EMG sensor
        if 'fquat' in file:
            new_file_name += 'imu_'
        elif 'emg' in file:
            new_file_name += 'emg_'

        # list what type of test is reported in this CSV file
        if 'stim_walk' in file:
            new_file_name += 'stimwalk'
        elif 'unstimulated_walk' in file:
            new_file_name += 'unstimwalk'
        elif 'ready_unassisted' in file:
            new_file_name += 'unassisted'
        elif 'assisted' in file:
            new_file_name += 'assisted'
        elif 'standby' in file:
            new_file_name += 'standby0'
            if file_num == '000':
                new_file_name += '1'
            elif file_num == '002':
                new_file_name += '2'
    
        # rename the file
        new_file_name = new_file_name + '.csv'
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))
    print('finished renaming: ' + str(folder_path))

def rename_files(parentdir):
    '''
    Renames all files in the MS Pilot Study directory that was downloaded

    Inputs:
        - parentdir (str): The path to the parent directory
    '''
    # get Cionic's path that they place the study's files in 
    study_path = Path(parentdir) / 'cionic/MSPilotCSU'

    # rename files for every directory
    for protocol_folder in study_path.iterdir():
        if protocol_folder.is_dir():
            for collection in protocol_folder.iterdir():
                name_file(collection)

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

def get_csvs(orgid, studyid, protoids, outdir, csvs):
    '''
    Gets all the CSV files for the specified study

    Inputs:
        - orgid (str): The organization ID
        - studyid (str): The name of the study
        - protoids (list): List of protocol names
        - outdir (str): The path to place the folders
        - csvs (list): Which CSVs to retrieve (e.g. ['emg', 'fquat'] would retrieve both EMG and IMU data)
    '''
    # specify the token path and the organization
    tokenpath = 'token.json'
    orgs = dl.cionic.auth(tokenpath=tokenpath)
    if orgid == None:
        for (i,o) in enumerate(orgs):
            print(f"{i} : {o['shortname']}")
        choice = int(input("Choose an org\n"))
        orgid = orgs[choice]['shortname']

    # select the study
    studies = dl.cionic.get_cionic(f"{orgid}/studies")
    sxid = 0
    for (i,s) in enumerate(studies):
        if studyid == s['shortname']:
            sxid = s['xid']

    # go through all the protocols that were specified
    protocols = dl.cionic.get_cionic(f"{orgid}/protocols?sxid={sxid}")
    for protoid in protoids:
        pxid = 0
        for (i, p) in enumerate(protocols):
            if protoid == p['shortname']:
                pxid = p['xid']

        # fetch collections
        collections = dl.cionic.get_cionic(f"{orgid}/collections?protoxid={pxid}")
        collections = sorted(collections, key=lambda collection: -collection['created_ts'])
            
        # download and parse
        fileroot = f"{outdir}/{orgid}/{studyid}/{protoid}"
        urlroot = f"{orgid}/collections"
        nameroot = f"{orgid}_{studyid}"
        dl.load_collections(collections, urlroot, fileroot, nameroot, False, csvs)

def main():
    orgid = 'cionic' # organization name
    studyid = 'MSPilotCSU' # study name
    protoids = ['6MWT', 'T25FW'] # protocols: 6-minute walk test and timed 25-foot walk
    csvs = ['emg', 'fquat'] # which sensor data to retrieve (getting both EMG and IMU)
    data_dir = './source_data' # directory that the study collections are stored in
    out_dir = './data'
    subjects_df = pd.read_csv('./SubjectToCollection.csv') # the dataframe linking subjects to collections

    # download all the relevant CSV files
    get_csvs(orgid, studyid, protoids, outdir=data_dir, csvs=csvs)

    # rename the downloaded files to more meaningful names
    rename_files(data_dir)

    # Move the files to the folder structure to be uploaded
    parse_data(data_dir, out_dir=out_dir, subjects_df=subjects_df)
            
 
if __name__ == '__main__':
    main()