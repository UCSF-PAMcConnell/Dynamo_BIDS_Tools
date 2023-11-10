import scipy.io
import pandas as pd
import argparse
import os
import glob
import logging
import re
import json

# Configuring logging to display informational messages and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load MATLAB data from the provided file path and extract relevant fields.
def load_matlab_data(matlab_file_path):
    """
    Parameters:
    matlab_file_path (str): Path to the MATLAB file.
    Returns:
    tuple: A tuple containing trial events and block data, or (None, None) if an error occurs.
    """
    try:
        mat_data = scipy.io.loadmat(matlab_file_path)
        trial_events = mat_data['trialEvents']
        block_data = mat_data['blockData'][0, 0]
        return trial_events, block_data
    except KeyError as e:
        logging.error(f"Missing necessary key in MATLAB file {matlab_file_path}: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading MATLAB file {matlab_file_path}: {e}")
        return None, None

# Format the MATLAB data into a pandas DataFrame suitable for BIDS .tsv.
def format_data_for_bids(trial_events, block_data):
    """
    Parameters:
    trial_events: Trial events data extracted from the MATLAB file.
    block_data: Block data extracted from the MATLAB file.
    Returns:
    DataFrame: Formatted data as a pandas DataFrame, or None if an error occurs.
    """
    try:
        onset_times_ms = trial_events['trialStart'][0, 0].flatten()
        onset_times_sec = onset_times_ms / 1000
        durations_sec = block_data['trialDuration'][0, 0].flatten()
        trial_types_ind = block_data['isSequenceTrial'][0, 0].flatten()
        trial_types = ['sequence' if ind == 1 else 'random' for ind in trial_types_ind]
      
        data = pd.DataFrame({
            'onset': onset_times_sec,
            'duration': durations_sec,
            'trial_type': trial_types
        })
        return data
    except KeyError as e:
        logging.error(f"Missing necessary key in data structure: {e}")
        return None
    except Exception as e:
        logging.error(f"Error formatting data for BIDS: {e}")
        return None

# Create a JSON sidecar file for a given BIDS .tsv file.    
def create_json_sidecar(tsv_file_path, data_frame):
    """
    Parameters:
    tsv_file_path (str): Path to the .tsv file.
    data_frame (DataFrame): The DataFrame containing the data.
    """
    json_file_path = tsv_file_path.replace('.tsv', '.json')
    description = {
        "onset": {
            "LongName": "Event onset time",
            "Description": "Time of the event measured from the beginning of the acquisition of the first volume in the corresponding task imaging data file, in seconds.",
            "Units": "seconds"
        },
        "duration": {
            "LongName": "Event duration",
            "Description": "Duration of the event, in seconds.",
            "Units": "seconds"
        },
        "trial_type": {
            "LongName": "Event category",
            "Description": "Blue circle fills one of four target levels, correct response precisely hits target with yellow circle, holds for the duration of the trial, and releases when blue circle returns to base position.",
            "Levels": {
                "sequence": "A sequence trial",
                "random": "A random trial"
            }
        }
    }

    try:
            with open(json_file_path, 'w') as json_file:
                json.dump(description, json_file, indent=4)
            logging.info(f"JSON sidecar file created at {json_file_path}")
    except Exception as e:
        logging.error(f"Error creating JSON sidecar file: {e}")

# Save the formatted DataFrame as a .tsv file and create a corresponding JSON sidecar file.
def save_as_tsv(data, output_path):
    """
    Parameters:
    data (DataFrame): The data to be saved.
    output_path (str): Path where the .tsv file will be saved.
    """
    if data is not None:
        try:
            directory = os.path.dirname(output_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            data.to_csv(output_path, sep='\t', index=False)
            create_json_sidecar(output_path, data)
        except Exception as e:
            logging.error(f"Error saving data to {output_path}: {e}")

# Main function
if __name__ == "__main__":
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process behavioral files and convert to .tsv in BIDS format.')
    parser.add_argument('matlab_dir', type=str, help='Directory containing the MATLAB files.')
    parser.add_argument('bids_root_dir', type=str, help='Root directory of the BIDS dataset where .tsv files will be saved.')
    args = parser.parse_args()
    
    # Define the order in which runs should appear
    run_order = [
        ("localizer_run1", "run-00"),
        ("localizer_run2", "run-07")
    ] + [("learningSession", f"run-{i:02d}") for i in range(1, 7)]

    # Defining specific run_ids for localizer runs
    localizer_run_ids = ["run-00", "run-07"]
    
    # Get all MATLAB files from the specified directory
    matlab_files = glob.glob(os.path.join(args.matlab_dir, "LRN*_*.mat"))

    for matlab_file_path in sorted(matlab_files):  # Processing files in alphabetical order
        try:
            filename = os.path.basename(matlab_file_path).rstrip('.mat')
            
            # Differentiating between localizer and learningSession runs
            if "localizer" in filename:
                run_id = localizer_run_ids.pop(0) if localizer_run_ids else None
            else:
                match = re.search(r'run(\d+)', filename)
                run_id = f"run-{int(match.group(1)):02d}" if match else None
            
            if run_id is None:
                logging.warning(f"Could not extract run information from filename: {filename}")
                continue
            
            # Load and format MATLAB data
            trial_events, block_data = load_matlab_data(matlab_file_path)
            if trial_events is None or block_data is None:
                logging.error(f"Failed to load data from {matlab_file_path}")
                continue

            formatted_data = format_data_for_bids(trial_events, block_data)
            if formatted_data is None:
                logging.error(f"Failed to format data for BIDS from {matlab_file_path}")
                continue
            
            # Construct output file path
            filename_parts = filename.split('_')
            subject_id = filename_parts[0]
            output_path = os.path.join(args.bids_root_dir, f"sub-{subject_id}", 'ses-2', 'func',
                                       f"sub-{subject_id}_ses-2_task-learn_{run_id}_events.tsv")
            
            # Save formatted data as .tsv with .json sidecar
            save_as_tsv(formatted_data, output_path)
            logging.info(f"Processed {matlab_file_path} and saved .tsv file to: {output_path}")
        except Exception as e:
            logging.error(f"An error occurred while processing {matlab_file_path}: {e}")