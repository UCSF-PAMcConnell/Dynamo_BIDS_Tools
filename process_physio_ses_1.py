import os
import re
import logging
import argparse
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import gzip

# Configure logging
logging.basicConfig(
    filename='process_physio.log',
    filemode='w', # a to append, w to overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Helper functions

# Extract the subject and session IDs from the physio_root_dir path
def extract_subject_session(physio_root_dir):
    """
    Parameters:
    - physio_root_dir: str, the directory path that includes subject and session information.
    Returns:
    - subject_id: str, the extracted subject ID
    - session_id: str, the extracted session ID
    """
    # Normalize the path to remove any trailing slashes for consistency
    physio_root_dir = os.path.normpath(physio_root_dir)

    # The pattern looks for 'sub-' followed by any characters until a slash, and similar for 'ses-'
    match = re.search(r'(sub-[^/]+)/(ses-[^/]+)', physio_root_dir)
    if not match:
        raise ValueError(f"Unable to extract subject_id and session_id from path: {physio_root_dir}")
    
    subject_id, session_id = match.groups()

    # Set up log to print the extracted IDs
    logging.info(f"Subject ID: {subject_id}, Session ID: {session_id}")
    return subject_id, session_id

# Loads the physio .mat file and extracts labels, data, and units
def load_mat_file(mat_file_path):
    """
    Parameters:
    - mat_file_path: str, path to the .mat file
    Returns:
    - labels: array, names of the physiological data channels
    - data: array, physiological data
    - units: array, units for each channel of physiological data
    """
    
    # Check if the file exists
    if not os.path.isfile(mat_file_path):
        logging.error(f"MAT file does not exist at {mat_file_path}")
        raise FileNotFoundError(f"No MAT file found at the specified path: {mat_file_path}")
    
    try:
        # Attempt to load the .mat file
        logging.info(f"Loading MAT file from {mat_file_path}")
        mat_contents = sio.loadmat(mat_file_path)
        
        # Verify that required keys are in the loaded .mat file
        required_keys = ['labels', 'data', 'units']
        if not all(key in mat_contents for key in required_keys):
            logging.error(f"MAT file at {mat_file_path} is missing one of the required keys: {required_keys}")
            raise KeyError(f"MAT file at {mat_file_path} is missing required keys")
        
        # Extract labels, data, and units
        labels = mat_contents['labels'].flatten()  # Flatten in case it's a 2D array
        data = mat_contents['data']
        units = mat_contents['units'].flatten()  # Flatten in case it's a 2D array
        
        # Log the labels and units for error checking
        logging.info(f"Labels extracted from MAT file: {labels}")
        logging.info(f"Units extracted from MAT file: {units}")
        logging.info(f"Successfully loaded MAT file from {mat_file_path}")
        
    except Exception as e:
        # Log the exception and re-raise to handle it upstream
        logging.error(f"Failed to load MAT file from {mat_file_path}: {e}")
        raise
    
    return labels, data, units

# Renames channels according to BIDS convention
def rename_channels(labels):
    """
    Parameters:
    - labels: array, original names of the physiological data channels
    Returns:
    - bids_labels: dict, mapping from original labels to BIDS-compliant labels
    """
    logging.info("Renaming channels according to BIDS conventions")
    
    # Define the mapping from original labels to BIDS labels
    original_label_mapping = {
        'ECG': 'cardiac',
        'RSP': 'respiratory',
        'EDA': 'eda',
        'Trigger': 'trigger',
        'PPG': 'ppg',  # Only if exists
    }

    # Initialize an empty dictionary and list to store the renamed labels
    bids_labels_dictionary = {}
    bids_labels_list = []

    # Iterate through the original labels to rename them in dictionary
    for label in labels:
        # Skip any labels for digital inputs
        if 'Digital input' in label:
            continue
        
        # Check and rename the label if it matches one of the keys in original_label_mapping
        for original, bids in original_label_mapping.items():
            if original in label:
                bids_labels_dictionary[label] = bids
                bids_labels_list.append(bids)
                break
        else:
            logging.warning(f"Label '{label}' does not match any BIDS convention and will be omitted.")

    # Debug log to print the renamed labels in the dictionary and the list
    logging.info(f"BIDS labels dictionary mapping: {bids_labels_dictionary}")
    logging.info(f"BIDS labels list after renaming: {bids_labels_list}")
    
    return bids_labels_dictionary, bids_labels_list

def find_runs(triggers, num_volumes):
    # Identifies the start and end indices of each run based on MR triggers and number of volumes
    logging.info("Identifying runs based on triggers and number of volumes")
    runs = []
    # Logic to identify runs...
    return runs


def segment_data(data, runs, sampling_rate):
    # Segments the physiological data based on the identified runs
    logging.info("Segmenting data into runs")
    segmented_data = []
    # Logic to segment data...
    return segmented_data


def write_output_files(segmented_data, metadata, output_dir):
    # Writes the output .tsv.gz and .json files for each run
    logging.info(f"Writing output files to {output_dir}")
    # Logic to write files...


def plot_runs(data, runs, output_file):
    # Plots the physiological data for all runs and saves the figure
    logging.info(f"Plotting runs and saving to {output_file}")
    # Logic to plot data...
    plt.savefig(output_file)


def extract_metadata_from_json(json_file_path):
    # Extracts necessary metadata from the .json file associated with each run
    logging.info(f"Extracting metadata from {json_file_path}")
    with open(json_file_path, 'r') as file:
        metadata = json.load(file)
    return metadata


# Main function to orchestrate the conversion process
def main(physio_root_dir, bids_root_dir):
    # Main logic here
    try:

        # Extract subject_id and session_id from the physio_root_dir path
        subject_id, session_id = extract_subject_session(physio_root_dir)
       
        # Construct the path to the .mat file using the naming convention
        mat_file_name = f"{subject_id}_{session_id}_task-rest_physio.mat"
        mat_file_path = os.path.join(physio_root_dir, mat_file_name)

        # Load .mat file
        labels, data, units = load_mat_file(mat_file_path)
        
        # Rename channels based on dynamic labels from data
        bids_labels = rename_channels(labels)
        

        all_runs_data = []  # To store data for all runs
        runs = [] # To store data for each run

        # Loop through each run directory in BIDS format
        # Iterate over the runs
        for run_idx in range(1, 5):  # Assuming there are 4 runs as specified
            run_id = f"run-{run_idx:02d}"
            
            # Construct the path to the run's .json file
            json_file_name = f"{subject_id}_{session_id}_task-rest_{run_id}_bold.json"
            json_file_path = os.path.join(bids_root_dir, subject_id, session_id, 'func', json_file_name)
            
            # Extract metadata from the run's .json file
            run_metadata = extract_metadata_from_json(json_file_path)
            
            # Find the runs in the data
            runs = find_runs(data, run_metadata['NumVolumes'])
            
            # Segment the data based on the runs
            segmented_data = segment_data(data, runs, run_metadata['SamplingFrequency'])
            
            # all_runs_data.append(segmented_data)  # Accumulate segmented data for plotting (?)
            
            # Write the output files for this run
            output_dir = os.path.join(bids_root_dir, subject_id, session_id, 'func')
            write_output_files(segmented_data, run_metadata, output_dir)

        # After processing all runs, plot the physiological data to verify alignment
        plot_file = f"{physio_root_dir}/{subject_id}_{session_id}_task-rest_all_runs_physio.png"
        plot_runs(data, runs, plot_file)
        # plot_runs(all_runs_data, plot_file)  # Adjusted to pass all_runs_data (?)
        
        logging.info("Process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir)