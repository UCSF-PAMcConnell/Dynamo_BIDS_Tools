import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import argparse
import re
import matplotlib.pyplot as plt
import warnings
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    filename='process_physio.log',  # Log to this file
    filemode='a',  # Append to the log file, 'w' would overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
    level=logging.INFO  # Set logging level to INFO
)

# Extract subject and session IDs from the directory path
def extract_subject_session_ids(physio_root_dir): 
    # Log the attempt to extract subject and session IDs.
    logging.info(f'Extracting subject and session IDs from {physio_root_dir}')
    
    # Extract subject and session IDs from the directory path
    pattern = r'sub-(?P<subject_id>\w+)/ses-(?P<session_id>\w+)/physio'
    match = re.search(pattern, physio_root_dir)
    if not match:
        error_msg = "Unable to extract subject_id and session_id from the physio_root_dir."
        logging.error(error_msg)
        raise ValueError(error_msg)
    subject_id, session_id = match.groups()
    
    # Log successful extraction of subject and session IDs.
    logging.info(f"Extracted subject ID: {subject_id}, session ID: {session_id}.")
    
    # Return the subject and session IDs
    print(f"Extracted subject ID: {subject_id}, session ID: {session_id}.")
    return subject_id, session_id

def locate_json_files(bids_root_dir, subject_id, session_id):
    # Log the attempt to locate JSON files.
    logging.info(f"Locating JSON files in {bids_root_dir}.")

    # Locate the functional BIDS directory containing the JSON files
    func_dir = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func')
    logging.info(f"Searching for JSON files in {func_dir}.")
    
    # Return the list of JSON files
    json_files = sorted([f for f in os.listdir(func_dir) if f.endswith('_bold.json')])
    return json_files, func_dir

# Define a function to load metadata from a JSON file
def load_json_metadata(bids_root_dir, subject_id, session_id, json_files):
    # Initiate an empty list to hold all metadata
    all_metadata = []
    
    for json_file in json_files:
        # Define the json path for each file
        json_path = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func', json_file)

        # Log the attempt to load JSON metadata.
        logging.info(f'Loading JSON metadata from {json_path}')

        try:
            # Open the JSON file and load its contents.
            with open(json_path, 'r') as file:
                metadata = json.load(file)
                all_metadata.append(metadata)
                # Log successful loading of metadata.
                logging.info(f'Successfully loaded metadata from {json_path}')
        except FileNotFoundError:
            # Log an error if the file is not found and raise the exception to the caller.
            logging.error(f'File not found: {json_path}', exc_info=True)
            raise
        except json.JSONDecodeError:
            # Log an error if the JSON file is not properly formatted and raise the exception to the caller.
            logging.error(f'Invalid JSON format in file: {json_path}', exc_info=True)
            raise
        except Exception as e:
            # Log any other exceptions that occur and raise the exception to the caller.
            logging.error(f'An unexpected error occurred while loading {json_path}: {e}', exc_info=True)
            raise

    # Log successful loading of all metadata.
    logging.info("Successfully loaded all metadata.")

    # Process each JSON file
    for json_file in json_files:
        # Extract number of volumes from the JSON file
        num_volumes = metadata['NumVolumes']
       
        # Extract repetition time from the JSON file
        tr = metadata['RepetitionTime']

        # Extract the run number from the json_file name
        run_number_match = re.search(r'run-(\d+)_bold.json', json_file)
        if run_number_match:
            run_number = int(run_number_match.group(1))
            # Check if the current run number exceeds the expected runs count
            if run_number > json_files.count(json_file):
                expected_runs_count = len(json_files)
                logging.warning(f"Run number {run_number} exceeds the expected count of {expected_runs_count}. Stopping.")
                break
        else:
            logging.error(f"Unable to extract run number from filename {json_file}")
            continue  # Skip this file and go to the next one

    # Return the metadata from all JSON files.
    return all_metadata, expected_runs_count, num_volumes, tr, json_files, json_path

def load_mat_file(physio_root_dir, subject_id, session_id):
    # Load the MATLAB file
    mat_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_physio.mat"
    mat_file_path = os.path.join(physio_root_dir, mat_file_name)
    
    # Log the attempt to load mat file.
    logging.info(f"Loading MATLAB file: {mat_file_path}.")
    try:
        mat_data = loadmat(mat_file_path)
    except FileNotFoundError:
        logging.error(f"The .mat file was not found at {mat_file_path}.")
        raise FileNotFoundError(f"The .mat file was not found at {mat_file_path}")
    
    # Validate contents of the MATLAB file
    if 'data' not in mat_data or 'labels' not in mat_data:
        error_msg = f"The .mat file {mat_file_name} does not contain 'data' and/or 'labels' keys."
        logging.error(error_msg)
        raise KeyError(error_msg)

    # Process the data and labels
    data, labels_struct = mat_data['data'], mat_data['labels']
    labels = [str(label.flat[0]) for label in labels_struct]
    logging.info("Data and labels loaded successfully.")
    
    print(labels)
    print(labels_struct)

    # Return the data and labels
    return data, labels, mat_file_name, mat_file_path, labels_struct

# Define a function to extract trigger points from MRI trigger channel data
def extract_trigger_points(mri_trigger_data, threshold=5):
    # Log the entry into the function and its parameters
    logging.info(f'Extracting trigger points with threshold: {threshold}')
    
    # Convert the MRI trigger data to binary based on the threshold
    # 1 if the data point is above the threshold, 0 otherwise
    triggers = (mri_trigger_data > threshold).astype(int)
    
    # Calculate the difference between successive trigger points to find rising edges
    diff_triggers = np.diff(triggers, prepend=0)
    
    # Find the indices of the start of trigger points (where the difference is 1)
    trigger_starts = np.where(diff_triggers == 1)[0]
    
    # Log the result before returning
    logging.info(f'Found {len(trigger_starts)} trigger points')
    
    # Check for empty trigger_starts, which would indicate no triggers were found
    if len(trigger_starts) == 0:
        logging.warning('No trigger points found. Check the threshold and input data.')

    return trigger_starts, triggers

def segment_data_into_runs(data, labels, metadata, trigger_starts):
    # Log the beginning of the segmentation process.
    logging.info('Segmenting data into runs...')
    
    runs = []
    for meta in metadata:
        num_volumes = meta['NumVolumes']
        tr = meta['RepetitionTime']
        sampling_rate = meta.get('SamplingRate', 5000)  # Use a default or meta provided value
        samples_per_volume = int(sampling_rate * tr)
        
        # Use trigger points to find the start of each run
        for start in trigger_starts:
            end_index = start + num_volumes * samples_per_volume
            if end_index > len(data):
                logging.warning("Insufficient data for the expected number of volumes.")
                break  # Break if we do not have enough data
            run_data = data[start:end_index]
            runs.append(run_data)
    
    # Log the end of the segmentation process.
    logging.info('Segmentation complete.')
    return runs

def main(physio_root_dir, bids_root_dir):
    logging.info("Starting the conversion process.")

    # Extract subject and session IDs
    subject_id, session_id = extract_subject_session_ids(physio_root_dir)

    # Load the .mat file
    data, labels, mat_file_name, mat_file_path, labels_struct = load_mat_file(physio_root_dir, subject_id, session_id)

    # Load metadata from the JSON files
    json_files, func_dir = locate_json_files(bids_root_dir, subject_id, session_id)
    metadata, expected_runs_count, num_volumes, tr = load_json_metadata(bids_root_dir, subject_id, session_id, json_files)
    
    # Extract trigger points
    trigger_label_index = labels.index(trigger_label)  # Find the index of the trigger label
    trigger_starts, triggers = extract_trigger_points(data[:, trigger_label_index])
    
    # Segment the data into runs
    segments = segment_data_into_runs(data, labels, num_volumes, tr, sampling_rate)

    # Save the segmented data to BIDS format


    # Save the segmented data to TSV and JSON files


    # Log the end of the conversion process.
    logging.info("Conversion process complete.")


# Command-line interface setup
if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat files.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset where the .json files are located.")
    args = parser.parse_args()

    # Call the main function
    main(args.physio_root_dir, args.bids_root_dir)