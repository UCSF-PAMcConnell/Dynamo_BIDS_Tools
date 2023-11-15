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

# Extract the subject and session IDs from the matlab_root_dir path
def extract_subject_session(matlab_root_dir):
    """
    Parameters:
    - matlab_root_dir: str, the directory path that includes subject and session information.
    Returns:
    - subject_id: str, the extracted subject ID
    - session_id: str, the extracted session ID
    """
    # Normalize the path to remove any trailing slashes for consistency
    matlab_root_dir = os.path.normpath(matlab_root_dir)

    # The pattern looks for 'sub-' followed by any characters until a slash, and similar for 'ses-'
    match = re.search(r'(sub-[^/]+)/(ses-[^/]+)', matlab_root_dir)
    if not match:
        raise ValueError("Unable to extract subject_id and session_id from path: %s", matlab_root_dir)
    
    subject_id, session_id = match.groups()

    # Set up log to print the extracted IDs
    return subject_id, session_id

# Check the MATLAB files and TSV files in the specified directories.
def check_files(matlab_root_dir, output_path, expected_mat_file_size_range_mb):
    """
    Check the MATLAB files and TSV files in the specified directories.

    Parameters:
    - matlab_root_dir (str): Directory containing the MATLAB files.
    - output_path (str): Directory where the TSV event files are expected to be saved.
    - expected_mat_file_size_range_mb (tuple): A tuple containing the minimum and maximum expected size of MATLAB files in megabytes.

    Returns:
    - bool: True if all checks pass, False otherwise.
    """

    # Convert MB to bytes for file size comparison
    min_size_bytes, max_size_bytes = [size * 1024 * 1024 for size in expected_mat_file_size_range_mb]

    # Check for exactly 8 MATLAB files
    mat_files = [f for f in os.listdir(matlab_root_dir) if f.endswith('.mat') and "processedData" not in f]
    if len(mat_files) != 8:
        logging.error(f"Incorrect number of MATLAB files in {matlab_root_dir}. Expected 8, found {len(mat_files)}.")
        return False

    # Check each MATLAB file size
    for file in mat_files:
        file_path = os.path.join(matlab_root_dir, file)
        file_size = os.path.getsize(file_path)
        if not (min_size_bytes <= file_size <= max_size_bytes):
            logging.error(f"MATLAB file {file} size is not within the expected range.")
            return False

    # Check that there are not already 8 TSV files in the output directory
    tsv_files = [f for f in os.listdir(output_path) if f.endswith('.tsv')]
    if len(tsv_files) >= 8:
        logging.error(f"Found 8 or more TSV files in {output_path}, indicating processing may already be complete.")
        return False

    return True

# Load MATLAB data from the provided file path and extract relevant fields.
def load_matlab_data(matlab_file_path):
    """
    Load MATLAB data from the specified file path and extract relevant task event and block data.

    This function reads a MATLAB file and extracts the 'trialEvents' and 'blockData' arrays, 
    which are expected to be present in the file. 'trialEvents' typically contain timing 
    and type information about different trials in a task, while 'blockData' may include 
    more detailed information about each block or condition in the task.

    Parameters:
    - matlab_file_path (str): Path to the MATLAB file.

    Returns:
    - tuple: A tuple containing 'trialEvents' and 'blockData'. Returns (None, None) if an error occurs.

    Raises:
    - KeyError: If the required keys ('trialEvents', 'blockData') are not found in the MATLAB file.
    - IOError: If there is an issue reading the file.
    - Exception: For any other exceptions during file loading.

    Example Usage:
    trial_events, block_data = load_matlab_data('/path/to/matlab_file.mat')
    """

    try:
        # Attempt to load the MATLAB file using scipy.io
        mat_data = scipy.io.loadmat(matlab_file_path)
        
        # Extract 'trialEvents' and 'blockData' from the loaded data
        trial_events = mat_data['trialEvents']
        block_data = mat_data['blockData'][0, 0]

        logging.info(f"Successfully loaded data from {matlab_file_path}")
        return trial_events, block_data

    except KeyError as e:
        # Log and handle missing key errors
        logging.error(f"Missing necessary key in MATLAB file {matlab_file_path}: {e}")
        return None, None

    except IOError as e:
        # Log and handle input/output errors
        logging.error(f"IOError encountered while loading MATLAB file {matlab_file_path}: {e}")
        return None, None

    except Exception as e:
        # Log and handle any other exceptions
        logging.error(f"Unexpected error while loading MATLAB file {matlab_file_path}: {e}")
        return None, None

# Format the MATLAB data into a pandas DataFrame suitable for BIDS .tsv.
def format_data_for_bids(trial_events, block_data):
    """
    Format the MATLAB data into a pandas DataFrame suitable for BIDS-compliant TSV files.

    This function takes trial events and block data from MATLAB file structures and formats them 
    into a pandas DataFrame. The DataFrame includes onset times, durations, and trial types, 
    making it suitable for conversion to a BIDS-compliant TSV file.

    Parameters:
    - trial_events: A structured array containing trial events data extracted from the MATLAB file.
    - block_data: A structured array containing block data extracted from the MATLAB file.

    Returns:
    - DataFrame: A pandas DataFrame formatted for BIDS, or None if an error occurs.

    Raises:
    - KeyError: If the required keys are not found in the input data structures.
    - Exception: For any other exceptions during data formatting.

    Example Usage:
    formatted_data = format_data_for_bids(trial_events, block_data)
    """

    try:
        # Extract onset times, convert from milliseconds to seconds
        onset_times_ms = trial_events['trialStart'][0, 0].flatten()
        onset_times_sec = onset_times_ms / 1000

        # Extract trial durations
        durations_sec = block_data['trialDuration'][0, 0].flatten()

        # Extract trial types and map them to descriptive labels
        trial_types_ind = block_data['isSequenceTrial'][0, 0].flatten()
        trial_types = ['sequence' if ind == 1 else 'random' for ind in trial_types_ind]

        # Create a pandas DataFrame with the formatted data
        data = pd.DataFrame({
            'onset': onset_times_sec,
            'duration': durations_sec,
            'trial_type': trial_types
        })

        logging.info("Data successfully formatted for BIDS")
        return data

    except KeyError as e:
        # Log and handle missing key errors
        logging.error(f"Missing necessary key in data structure for BIDS formatting: {e}")
        return None

    except Exception as e:
        # Log and handle any other exceptions
        logging.error(f"Unexpected error while formatting data for BIDS: {e}")
        return None

# Create a JSON sidecar file for a given BIDS-compliant TSV file.
def create_json_sidecar(tsv_file_path, data_frame):
    """
    Create a JSON sidecar file for a given BIDS-compliant TSV file.

    This function generates a JSON sidecar file containing metadata descriptions
    for each column in the provided TSV file. The metadata includes details such as 
    the name, description, and units for each column. This sidecar file is essential
    for BIDS compliance, as it provides contextual information about the data in the TSV file.

    Parameters:
    - tsv_file_path (str): Path to the .tsv file.
    - data_frame (DataFrame): The DataFrame containing the data. This is used to validate 
                              the presence of expected columns.

    The function assumes the presence of 'onset', 'duration', and 'trial_type' columns 
    in the DataFrame and describes these columns in the JSON file.

    Raises:
    - IOError: If there's an issue writing the JSON file.
    - Exception: For any other exceptions during the JSON file creation.

    Example Usage:
    create_json_sidecar('/path/to/data.tsv', data_frame)
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
            "Description": "Type of the trial, categorized as 'sequence' or 'random'.",
            "Levels": {
                "sequence": "A sequence trial",
                "random": "A random trial"
            }
        }
    }

    try:
        # Write the description dictionary to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(description, json_file, indent=4)
        logging.info(f"JSON sidecar file created at {json_file_path}")

    except IOError as e:
        # Log and handle input/output errors
        logging.error(f"IOError while creating JSON sidecar file {json_file_path}: {e}")

    except Exception as e:
        # Log and handle any other exceptions
        logging.error(f"Unexpected error while creating JSON sidecar file {json_file_path}: {e}")

# Save the formatted DataFrame as a .tsv file and create a corresponding JSON sidecar file.
def save_as_tsv(data, output_path):
    """
    Save the formatted DataFrame as a .tsv (Tab-Separated Values) file and create a corresponding JSON sidecar file.

    This function writes a provided pandas DataFrame to a TSV file at the specified output path.
    It also creates a JSON sidecar file that includes metadata descriptions for each column in the TSV file.
    The creation of both the TSV and JSON files is essential for compliance with BIDS (Brain Imaging Data Structure) standards.

    Parameters:
    - data (DataFrame): The pandas DataFrame containing formatted data to be saved.
    - output_path (str): The file path where the .tsv file will be saved. The function automatically 
                         determines the path for the JSON sidecar file based on this path.

    The function checks if the output directory exists and creates it if necessary. It then writes the DataFrame to
    a TSV file and calls the `create_json_sidecar` function to generate the corresponding JSON file.

    Raises:
    - Exception: If any error occurs during the saving process or while creating the JSON sidecar file.

    Example Usage:
    save_as_tsv(formatted_data, '/path/to/output.tsv')
    """

    if data is None:
        logging.error("No data provided to save as TSV.")
        return

    try:
        # Ensure the directory for the output file exists
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the DataFrame to a TSV file
        data.to_csv(output_path, sep='\t', index=False)
        logging.info(f"TSV file saved successfully at {output_path}")

        # Create the corresponding JSON sidecar file
        create_json_sidecar(output_path, data)

    except Exception as e:
        # Log any exceptions that occur during the saving process
        logging.error(f"Error saving data to {output_path}: {e}")

def matches_pattern(filename, pattern):
    return all(part in filename for part in pattern.split())


# Main function to process all MATLAB files in the specified directory and save them as .tsv files and JSON sidecar files.
def main(matlab_root_dir, bids_root_dir):
    """
    Process all MATLAB files in the specified directory and save them as BIDS-compliant .tsv files and JSON sidecar files.

    This function iterates through all MATLAB files in the given directory, formats the extracted data for BIDS compliance,
    and saves each as a .tsv file accompanied by a JSON sidecar file in the specified BIDS dataset directory. 
    It handles localizer and learningSession runs separately and assigns appropriate BIDS run identifiers.

    Parameters:
    - matlab_root_dir (str): Directory containing the MATLAB files.
    - bids_root_dir (str): Root directory of the BIDS dataset where .tsv files and JSON sidecar files will be saved.

    The function assumes a specific naming convention for MATLAB files (LRN*_*.mat) and extracts run identifiers 
    from filenames. It logs detailed information and errors encountered during processing.

    Raises:
    - Exception: For any errors that occur during the processing of each MATLAB file.

    Example Usage:
    main('/path/to/matlab_root_dir', '/path/to/bids_root_dir')
    """
    # Extract subject and session IDs from the path
    subject_id, session_id = extract_subject_session(matlab_root_dir)
    
    # Example values for expected MATLAB file sizes in megabytes
    expected_mat_file_size_range_mb = (8, 21)

    # Construct the output path based on subject_id and session_id
    output_path = os.path.join(bids_root_dir, subject_id, session_id, 'func') 

    # Check MATLAB and TSV files before processing
    if not check_files(matlab_root_dir, output_path, expected_mat_file_size_range_mb):
        print(f"Initial file check failed. Exiting script.")
        return # Exit the script if file check fails.
    
    # Setup logging after extracting subject_id and session_id.
    setup_logging(subject_id, session_id, bids_root_dir)
    logging.info("Processing subject: %s, session: %s", subject_id, session_id)

    # Define the order and identifiers for different types of runs (multiple methods provided for debugging).
    subject_id_without_prefix = subject_id.replace('sub-', '')  # Remove 'sub-' prefix

    run_order = [
        (f"{subject_id_without_prefix}_localizer_run1_*_NS.mat", "run-00"),
        (f"{subject_id_without_prefix}_localizer_run2_*_NS.mat", "run-07"),
        (f"{subject_id_without_prefix}_learningSession_*_run1_*_NS.mat", "run-01"),
        (f"{subject_id_without_prefix}_learningSession_*_run2_*_NS.mat", "run-02"),
        (f"{subject_id_without_prefix}_learningSession_*_run3_*_NS.mat", "run-03"),
        (f"{subject_id_without_prefix}_learningSession_*_run4_*_NS.mat", "run-04"),
        (f"{subject_id_without_prefix}_learningSession_*_run5_*_NS.mat", "run-05"),
        (f"{subject_id_without_prefix}_learningSession_*_run6_*_NS.mat", "run-06")
    ]
    
    matlab_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/sourcedata/sub-LRN017/ses-2/beh/preprocessed/')
    files = glob.glob(matlab_root_dir)
    print(f"Found files: {files}")
    try:
        # Process each run in the defined order
        for run_name, run_id in run_order:
            full_path = os.path.join(matlab_root_dir, run_name)
            logging.info(f"Full path being searched: {full_path}")
            
            # Construct the file pattern for MATLAB files
            matlab_file_pattern = f"{run_name}"
            logging.info(f"Searching for files: {matlab_file_pattern}")

            # Find all MATLAB files that match the current pattern
            matlab_files = glob.glob(os.path.join(matlab_root_dir, matlab_file_pattern))
            logging.info(f"Found files: {matlab_files}")

            # Process each MATLAB file for the current run
            for matlab_file_path in sorted(matlab_files):
                try:
                    # Extract file name
                    filename = os.path.basename(matlab_file_path).rstrip('.mat')
                    logging.info(f"Processing file: {filename} for run: {run_id}")

                    # Load and format MATLAB data
                    trial_events, block_data = load_matlab_data(matlab_file_path)
                    if trial_events is None or block_data is None:
                        logging.error(f"Error loading data from {matlab_file_path}")
                        continue  # Skip this file if loading fails

                    formatted_data = format_data_for_bids(trial_events, block_data)
                    if formatted_data is None:
                        logging.error(f"Error formatting data from {matlab_file_path}")
                        continue  # Skip this file if formatting fails

                    # Construct the output file path
                    output_path = os.path.join(bids_root_dir, f"{subject_id}", session_id, 'func',
                                            f"{subject_id}_{session_id}_task-learn_{run_id}_events.tsv")

                    # Save formatted data as TSV
                    save_as_tsv(formatted_data, output_path)
                    logging.info(f"Saved formatted data to: {output_path}")

                except Exception as e:
                    logging.error(f"Error processing {matlab_file_path}: {e}")
                    # Decide whether to raise an exception or continue with the next file
                    continue  # or 'raise' to stop processing

    except Exception as e:
        logging.error(f"Unexpected error during processing: {e}")
        raise  # Propagate the exception upwards

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
    main(args.matlab_root_dir, args.bids_root_dir)