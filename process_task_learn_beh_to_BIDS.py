"""
process_task_learn_beh_to_BIDS.py

Description:
This script processes behavioral data for Task fMRI studies from MATLAB files and converts them into BIDS-compliant TSV files. 
It loads MATLAB files, extracts necessary task event information, and formats the data into a structured DataFrame suitable for BIDS. 
Additionally, it generates accompanying JSON sidecar files with metadata. The script features robust error handling, logging, and is 
designed for easy integration into BIDS data processing pipelines.

Usage:
Invoke the script from the command line with the following format:
python process_task_learn_beh_to_BIDS.py <matlab_dir> <bids_root_dir>

Example usage:
python process_task_learn_beh_to_BIDS.py /path/to/matlab_dir /path/to/bids_root_dir

Author: PAMcConnell
Created on: 20231112
Last Modified: 20231112
Version: 1.0.0

License:
This software is released under the MIT License.

Dependencies:
- Python 3.12
- Python libraries: scipy, pandas, argparse, os, glob, logging, re, json, sys.
- MATLAB files containing task fMRI behavioral data.

Environment Setup:
- Ensure Python 3.12 is installed in your environment.
- Install required Python libraries using 'pip install scipy pandas'.
- Ensure that MATLAB files are structured correctly and accessible in the specified directory.

Change Log:
- 20231111: Initial release of the script with basic functionality for loading and processing MATLAB files.
"""

import scipy.io                   # for loading .mat files
import pandas as pd               # for loading .csv files
import argparse                   # for command line arguments
import os                         # for file and directory operations
import glob                       # for unix file pattern matching
import logging                    # for logging
import re                         # for regular expressions
import json                       # for json file operations
import sys                        # for sys.exit and sys.argv

# Set up logging for individual archive logs.
def setup_logging(subject_id, session_id, bids_root_dir):
    """
    Sets up logging for the script, creating log files in a specified directory.

    Parameters:
    - subject_id (str): The identifier for the subject.
    - session_id (str): The identifier for the session.
    - bids_root_dir (str): The root directory of the BIDS dataset.

    This function sets up a logging system that writes logs to both a file and the console. 
    The log file is named based on the subject ID, session ID, and the script name. 
    It's stored in a 'logs' directory within the 'doc' folder by subject ID, which is located at the same 
    level as the BIDS root directory.

    The logging level is set to INFO, meaning it captures all informational, warning, and error messages.

    Usage Example:
    setup_logging('sub-01', 'ses-1', '/path/to/bids_root_dir')
    """

    # Extract the base name of the script without the .py extension.
    script_name = os.path.basename(__file__).replace('.py', '')

    # Construct the log directory path within 'doc/logs'
    log_dir = os.path.join(os.path.dirname(bids_root_dir), 'doc', 'logs', script_name, subject_id)

    # Create the log directory if it doesn't exist.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the log file name using subject ID, session ID, and script name.
    log_file_name = f"{subject_id}_{session_id}_{script_name}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    # Configure file logging.
    logging.basicConfig(
        level=logging.INFO,
        # filename='process_physio_ses_2.log', # Uncomment this line to save log in script execution folder.
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w' # 'w' mode overwrites existing log file.
    )

    # If you also want to log to console.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Logging setup complete. Log file: {log_file_path}")

    return log_file_path

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

def main(matlab_root_dir, bids_root_dir):
    """
    Parameters:
    - matlab_root_dir (str): Directory containing the MATLAB files.
    - bids_root_dir (str): Root directory of the BIDS dataset where .tsv files will be saved.
    """
   # Define the order in which runs should appear
    run_order = [
        ("localizer_run1", "run-00"),
        ("localizer_run2", "run-07")
    ] + [("learningSession", f"run-{i:02d}") for i in range(1, 7)]

    # Defining specific run_ids for localizer runs
    localizer_run_ids = ["run-00", "run-07"]

    # Get all MATLAB files from the specified directory
    matlab_files = glob.glob(os.path.join(args.matlab_root_dir, "LRN*_*.mat"))

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

# Main function executed when the script is run from the command line.
if __name__ == "__main__":
    
   # Set up an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description='Process .mat behavioral files for TASK FMRI and convert to BIDS tsv.')

    # Add arguments to the parser.

    # The first argument is the root directory containing the DICOM directories.
    parser.add_argument('matlab_root_dir', type=str, help='Root directory containing the DICOM directories.')
    
    # The second argument is the root directory of the BIDS dataset.
    parser.add_argument('bids_root_dir', type=str, help='Root directory of the BIDS dataset.')

    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Call the main function with the parsed arguments.
    main(args.dicom_root_dir, args.bids_root_dir)