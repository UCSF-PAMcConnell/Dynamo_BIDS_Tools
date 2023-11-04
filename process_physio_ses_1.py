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

# Extracts required metadata from the .json file associated with each fMRI run.
def extract_metadata_from_json(json_file_path, processed_jsons):
    """
    Parameters:
    - json_file_path: str, path to the .json file
    - processed_jsons: set, a set of paths to already processed JSON files
    Returns:
    - run_metadata: dict, specific metadata required for processing
    """
    logging.info(f"Extracting metadata from {json_file_path}")

    # Skip processing if this file has already been processed
    if json_file_path in processed_jsons:
        logging.info(f"JSON file {json_file_path} has already been processed.")
        return None

    # Check if the file exists
    if not os.path.isfile(json_file_path):
        logging.error(f"JSON file does not exist at {json_file_path}")
        raise FileNotFoundError(f"No JSON file found at the specified path: {json_file_path}")

    try:
        # Attempt to open and read the JSON file
        with open(json_file_path, 'r') as file:
            metadata = json.load(file)
        
        # Extract only the required fields
        run_metadata = {
            'TaskName': metadata.get('TaskName'),
            'RepetitionTime': metadata.get('RepetitionTime'),
            'NumVolumes': metadata.get('NumVolumes')
        }

        # Check if all required fields were found
        if not all(run_metadata.values()):
            missing_fields = [key for key, value in run_metadata.items() if value is None]
            logging.error(f"Missing required metadata fields in {json_file_path}: {missing_fields}")
            raise ValueError(f"JSON file {json_file_path} is missing required fields: {missing_fields}")

        # Add this file to the set of processed JSON files
        processed_jsons.add(json_file_path)

        # Log the successful extraction of metadata
        logging.info(f"Successfully extracted metadata from {json_file_path}")
        
        # Log the extracted metadata
        logging.info(f"Successfully extracted metadata: {run_metadata}")

        # Check run_metadata type
        logging.info(f"run_metadata (type: {type(run_metadata)}): {run_metadata}")

    except json.JSONDecodeError as e:
        # Log an error if the JSON file is not properly formatted
        logging.error(f"Error decoding JSON from file {json_file_path}: {e}")
        raise
    
    return run_metadata, run_metadata['NumVolumes'], run_metadata['RepetitionTime'], run_metadata['TaskName']

# Identifies the start and end indices of each run based on MR triggers and number of volumes.
def find_runs(data, triggers, run_metadata, sampling_rate=5000, trigger_threshold=5):
    logging.info("Identifying runs based on sequences of triggers")
    
    trigger_indices = np.where(triggers > trigger_threshold)[0]
    runs = []
    volume_count = 0
    current_run_start_index = None

    # Variables for debug logging
    expected_interval = sampling_rate * run_metadata['RepetitionTime']
    trigger_intervals = []

    for i in range(len(trigger_indices)):
        # If we are at the start of the data or have just completed a run, look for a new run
        if volume_count == 0:
            current_run_start_index = trigger_indices[i]
            volume_count = 1  # Start counting volumes with the first trigger
        else:
            # Calculate the interval between the current and the previous trigger
            actual_interval = trigger_indices[i] - trigger_indices[i - 1]
            trigger_intervals.append(actual_interval)

            # Check if the current trigger is within an acceptable range of the expected interval
            if expected_interval * 0.8 <= actual_interval <= expected_interval * 1.2:
                volume_count += 1
                # If we have found the correct number of volumes, we have identified a run
                if volume_count == run_metadata['NumVolumes']:
                    runs.append({'start': current_run_start_index, 'end': trigger_indices[i-1]})
                    volume_count = 0  # Reset volume count for the next run
            else:
                # If the trigger is not in the expected range, reset and look for a new run
                volume_count = 1
                current_run_start_index = trigger_indices[i]

        # Log for debugging
        logging.info(f"Trigger index: {i}, Volume count: {volume_count}")

    logging.info(f"Trigger intervals: {trigger_intervals}")
    logging.info(f"Identified runs: {runs}")
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
        bids_labels_dictionary, _ = rename_channels(labels)
        
        # Log the labels
        logging.info(f"Labels: {labels}")
        
        # Log the bids_labels_dictionary to confirm the 'trigger' key exists
        logging.info(f"bids_labels_dictionary right before accessing 'trigger' key: {bids_labels_dictionary}")

        # Ensure that the original labels are used to find the index of the trigger channel
        # Find the original label for 'trigger' from the bids_labels_dictionary
        for original_label, bids_label in bids_labels_dictionary.items():
            if bids_label == 'trigger':
                trigger_original_label = original_label
                break

        trigger_channel_index = labels.tolist().index(trigger_original_label)
        logging.info(f"Trigger channel index: {trigger_channel_index}")
        
        trigger_channel_data = data[:, trigger_channel_index]
        logging.info(f"Shape of trigger channel data: {trigger_channel_data.shape}")

        processed_jsons = set()  # Initialize set to keep track of processed JSON files
        all_runs_data = []  # To store data for all runs
        runs_info = []  # To store metadata for each run
        
        # Loop through each run directory in BIDS format
        # Iterate over the runs
        for run_idx in range(1, 5):  # Assuming there are 4 runs as specified
            run_id = f"run-{run_idx:02d}"
            
            # Construct the path to the run's .json file
            json_file_name = f"{subject_id}_{session_id}_task-rest_{run_id}_bold.json"
            json_file_path = os.path.join(bids_root_dir, subject_id, session_id, 'func', json_file_name)
            
            # Extract metadata from the JSON file
            run_metadata, num_volumes, repetition_time, task_name = extract_metadata_from_json(json_file_path, processed_jsons)

            # Before calling find_runs, check if run_metadata is None
            if run_metadata is None:
                logging.error("run_metadata is None, skipping run.")
                continue  # Skip to the next iteration of the loop

            # Now add the checking code
            if not isinstance(run_metadata, dict) or 'RepetitionTime' not in run_metadata or 'NumVolumes' not in run_metadata:
                logging.error("run_metadata is not a dictionary or does not contain the required keys.")
                continue

            # Find the runs in the data
            # current_runs_info = find_runs(data, trigger_channel_data, run_metadata['NumVolumes'], run_metadata['RepetitionTime'], run_metadata['TaskName'], sampling_rate=sampling_rate, trigger_threshold=5)
            # Call find_runs with the extracted metadata
            current_runs_info = find_runs(data, trigger_channel_data, run_metadata, sampling_rate=5000, trigger_threshold=5)
            logging.info(f"Found {len(current_runs_info)} runs")
            
            # Log the shape of the data array
            logging.info(f"Shape of data array: {data.shape}")

            # Log the current run
            logging.info(f"Processing run {run_id}")

            # If run_metadata is None, it means this JSON has already been processed or is invalid
            if run_metadata is None:
                # Log a warning and skip to the next iteration
                logging.warning(f"Metadata for {json_file_name} could not be extracted.")
                continue  # Skip to the next iteration

            # Check if current_runs_info is a list and has at least one element
            if isinstance(current_runs_info, list) and len(current_runs_info) > 0:
                # Access the first run's information if needed, or loop over all runs
                first_run_info = current_runs_info[0]

                for current_run_info in current_runs_info:
                    # Log the start and end indices of the current run
                    logging.info(f"Segmenting run {run_id} from index {current_run_info['start']} to {current_run_info['end']}")
                    
                    # Segment the data based on the runs
                    segmented_data = segment_data(data, trigger_channel_data, run_metadata['NumVolumes'], run_metadata['RepetitionTime'], trigger_threshold=5, sampling_rate=5000)
                    
                    # Log the shape of the segmented data
                    logging.info(f"Segmented data shape for run {run_id}: {segmented_data.shape}")  

                    # Accumulate segmented data for plotting
                    all_runs_data.append(segmented_data)  
                
                    # Append current run metadata to runs_info
                    runs_info.append(current_run_info)

                    # Write the output files for this run
                    output_dir = os.path.join(bids_root_dir, subject_id, session_id, 'func')
                    write_output_files(segmented_data, run_metadata, output_dir)
                else:
                    # Handle the case where no runs are found or an unexpected return type is received
                    logging.error("No runs found or unexpected return type from find_runs function.")

        # After processing all runs, plot the physiological data to verify alignment
        plot_file = f"{physio_root_dir}/{subject_id}_{session_id}_task-rest_all_runs_physio.png"
        plot_runs(all_runs_data, runs_info, plot_file)
        
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