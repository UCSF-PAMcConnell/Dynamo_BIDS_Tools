"""
BIDS_process_physio_ses_1.py

Description:
This Python script automates the conversion of physiological data from MATLAB format into BIDS-compliant TSV files. 
It is designed for Resting-State fMRI studies, extracting physiological data, such as ECG, respiratory, and EDA signals. 
The script renames channels following BIDS conventions, segments data based on trigger points, and attaches relevant metadata. 
It also includes visualization features, generating plots for quality checks. 
Robust error handling, detailed logging, and a command-line interface are key aspects, ensuring ease of use in BIDS data processing pipelines.

Usage:
Invoke the script from the command line with the following format:
python BIDS_process_physio_ses_1.py <physio_dir> <bids_root_dir>

Example usage:

python BIDS_process_physio_ses_1.py <physio_root_dir> <bids_root_dir> [--cut_off_duration <duration_in_minutes>]

Where --cut_off_duration is an optional argument to specify the duration in minutes to cut off from the start of the plot.

Author: PAMcConnell
Created on: 20231113
Last Modified: 20231113
Version: 1.0.0

License:
This software is released under the MIT License.

Dependencies:
- Python 3.12
- Libraries: scipy, pandas, matplotlib, numpy, json, glob
- MATLAB files containing physiological data for Task fMRI studies.

Environment Setup:
- Ensure Python 3.12 and necessary libraries are installed.
- MATLAB files should be accessible in the specified directory, following a structured format.

Change Log:
- 20231113: Initial release with functions for data loading, processing, segmenting, output file writing, and plotting.

"""

import os                                               # for working with directories and files.
import re                                               # for regular expressions and string manipulation.          
import logging                                          # for logging progress and errors.
import argparse                                         # for parsing command-line arguments.
import scipy.io as sio                                  # for loading .mat files.
import numpy as np                                      # for numerical operations and arrays.
import pandas as pd                                     # for data manipulation and analysis.
import matplotlib.pyplot as plt                         # for plotting data and visualizations.
import json                                             # for handling JSON data.
import glob                                             # for finding files in directories.
from matplotlib.backends.backend_pdf import PdfPages    # for creating multipage PDFs with matplotlib plots.
from collections import OrderedDict                     # for creating ordered dictionaries. 
import sys                                              # for accessing system-specific parameters and functions.

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
        raise ValueError("Unable to extract subject_id and session_id from path: %s", physio_root_dir)
    
    subject_id, session_id = match.groups()

    # Set up log to print the extracted IDs
    return subject_id, session_id

# Check the MATLAB files and TSV files in the specified directories.
def check_files(physio_root_dir, output_path, expected_mat_file_size_range_mb):
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

    # Check for exactly 1 MATLAB files
    mat_files = [f for f in os.listdir(physio_root_dir) if f.endswith('.mat') and "processedData" not in f]
    if len(mat_files) != 1:
        logging.error(f"Incorrect number of MATLAB files in {physio_root_dir}. Expected 1, found {len(mat_files)}.")
        return False

    # Check each MATLAB file size
    for file in mat_files:
        file_path = os.path.join(physio_root_dir, file)
        file_size = os.path.getsize(file_path)
        if not (min_size_bytes <= file_size <= max_size_bytes):
            logging.error(f"MATLAB file {file} size is not within the expected range.")
            return False

    # Check that there are not already 8 TSV files in the output directory
    tsv_files = [f for f in os.listdir(output_path) if f.endswith('physio.tsv')]
    if len(tsv_files) >= 8:
        logging.error(f"Found 8 or more TSV files in {output_path}, indicating processing may already be complete.")
        return False

    return True

# Loads the physio .mat file and extracts labels, data, and units
def load_mat_file(mat_file_path):
    """
    Load a MATLAB (.mat) file and extract labels, data, and units for physiological data.

    Parameters:
    - mat_file_path (str): Absolute or relative path to the .mat file.

    Returns:
    - tuple: A tuple containing three elements:
        - labels (array): Names of the physiological data channels.
        - data (array): Physiological data.
        - units (array): Units for each channel of physiological data.

    Raises:
    - FileNotFoundError: If the .mat file does not exist at the specified path.
    - KeyError: If the .mat file is missing any of the required keys ('labels', 'data', 'units').
    - Exception: Any other exceptions encountered during file loading.

    Usage Example:
    labels, data, units = load_mat_file('/path/to/matfile.mat')

    Dependencies:
    - scipy.io for loading MATLAB files.
    - os for file existence checking.
    - logging for logging information and errors.
    """

   # Check if the file exists
    if not os.path.isfile(mat_file_path):
        error_msg = f"MAT file does not exist at {mat_file_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Attempt to load the .mat file
        logging.info("Loading MAT file from %s", mat_file_path)
        mat_contents = sio.loadmat(mat_file_path)
        
        # Verify that required keys are in the loaded .mat file
        required_keys = ['labels', 'data', 'units']
        if not all(key in mat_contents for key in required_keys):
            error_msg = f"MAT file at {mat_file_path} is missing required keys: {required_keys}"
            logging.error(error_msg)
            raise KeyError(error_msg)
        
        # Extract labels, data, and units
        labels = mat_contents['labels'].flatten()  # Flatten in case it's a 2D array
        data = mat_contents['data']
        logging.info("Data loaded successfully with shape: %s", data.shape)
        # logging.info("Type of 'data': %s", type(data))
        units = mat_contents['units'].flatten()  # Flatten in case it's a 2D array
        
        # Log the labels and units for error checking
        #logging.info("Labels extracted from MAT file: %s", labels)
        #logging.info("Units extracted from MAT file: %s", units)
        logging.info("Successfully loaded MAT file from %s", mat_file_path)
        
    except Exception as e:
        # Log the exception and re-raise to handle it upstream
        logging.error("Failed to load MAT file from %s: %s", mat_file_path, e)
        raise
    
    return labels, data, units

# Renames channels according to BIDS convention
def rename_channels(labels):
    """
    Renames physiological data channels according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - labels (list or array): Original names of the physiological data channels.

    Returns:
    - tuple: A tuple containing two elements:
        - bids_labels_dictionary (dict): A dictionary mapping from original labels to BIDS-compliant labels.
        - bids_labels_list (list): A list of BIDS-compliant labels.

    This function iterates through the provided original labels and renames them based on predefined BIDS conventions.
    It logs a warning for any label that does not match the BIDS naming convention and excludes it from the output.

    Usage Example:
    bids_labels_dict, bids_labels_list = rename_channels(['ECG', 'RSP', 'EDA'])

    Dependencies:
    - logging module for logging information, warnings, and errors.
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
            logging.warning("Label '%s' does not match any BIDS convention and will be omitted.", label)

    # Debug log to print the renamed labels in the dictionary and the list
    logging.info("BIDS labels dictionary mapping: %s", bids_labels_dictionary)
    logging.info("BIDS labels list after renaming: %s", bids_labels_list)
    
    return bids_labels_dictionary, bids_labels_list

#  Extracts metadata from a JSON file and the associated run_id.
def extract_metadata_from_json(json_file_path, processed_jsons):
    """
    Extracts specific metadata from a JSON file and returns the run identifier and metadata.

    Parameters:
    - json_file_path (str): Path to the JSON file.
    - processed_jsons (set): A set of already processed JSON file paths.

    Returns:
    - tuple: (run_id, run_metadata) where:
        - run_id (str): The identifier for the run extracted from the file name.
        - run_metadata (dict): Dictionary containing metadata for the run.

    Raises:
    - FileNotFoundError: If the JSON file does not exist at the specified path.
    - ValueError: If the run identifier is not found in the file name or required metadata fields are missing.
    - json.JSONDecodeError: If there is an error decoding the JSON file.

    Usage Example:
    run_id, metadata = extract_metadata_from_json('/path/to/file.json', processed_jsons)

    Dependencies:
    - json module for reading JSON files.
    - os module for file operations.
    - re module for regular expressions.
    - logging module for logging information and errors.
    """
    
    logging.info("Extracting metadata from %s", json_file_path)
    
     # Check if the JSON file has already been processed
    if json_file_path in processed_jsons:
        logging.info("JSON file %s has already been processed.", json_file_path)
        return None, None  # No new metadata to return

    # Verify the existence of the JSON file
    if not os.path.isfile(json_file_path):
        logging.error("JSON file does not exist at %s", json_file_path)
        raise FileNotFoundError(f"No JSON file found at the specified path: {json_file_path}")

    try:
        # Read and parse the JSON file
        with open(json_file_path, 'r') as file:
            metadata = json.load(file)

        # Extract run ID from the file name
        run_id_match = re.search(r'run-\d+', json_file_path)
        if not run_id_match:
            raise ValueError(f"Run identifier not found in JSON file name: {json_file_path}")
        run_id = run_id_match.group()

        # Extract required metadata fields
        required_fields = ['TaskName', 'RepetitionTime', 'NumVolumes']
        run_metadata = {field: metadata.get(field) for field in required_fields}
        if not all(run_metadata.values()):
            missing_fields = [key for key, value in run_metadata.items() if value is None]
            raise ValueError(f"JSON file {json_file_path} is missing required fields: {missing_fields}")

        # Add the file to the set of processed JSONs
        processed_jsons.add(json_file_path)
        logging.info(f"Successfully extracted metadata for {run_id}: {run_metadata}")

        return run_id, run_metadata

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {json_file_path}: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred while extracting metadata from {json_file_path}: {e}")
        raise

# Identifies potential starts of triggers based on the threshold and minimum number of consecutive points.
def extract_trigger_points(data, threshold, min_consecutive):
    """
    Identifies potential starts of triggers in MRI trigger channel data based on a specified threshold and 
    minimum number of consecutive data points.

    Parameters:
    - data (numpy array): The MRI trigger channel data.
    - threshold (float): The value above which a data point is considered as a trigger start.
    - min_consecutive (int): Minimum number of consecutive data points above the threshold required 
                              to consider a point as a valid trigger start.

    Returns:
    - list: A list of indices where potential trigger starts are identified.

    The function first converts the data to a binary sequence based on the threshold, then identifies changes 
    from 0 to 1 as potential trigger starts. It further checks these starts to ensure they have the specified 
    minimum number of consecutive points above the threshold.

    Usage Example:
    valid_trigger_starts = extract_trigger_points(data_array, 2.5, 4)

    Dependencies:
    - numpy for array operations.
    - logging module for logging information.
    """

    logging.info("Extracting trigger points with threshold: %s and minimum consecutive points: %s", threshold, min_consecutive)

    # Convert data to a binary sequence based on the threshold
    triggers = (data > threshold).astype(int)

    # Identify changes from 0 to 1 as potential trigger starts
    diff_triggers = np.diff(triggers, prepend=0)
    potential_trigger_starts = np.where(diff_triggers == 1)[0]

    valid_trigger_starts = []
    for start in potential_trigger_starts:
        # Check if the start has the required minimum number of consecutive points above the threshold
        if np.sum(triggers[start:start+min_consecutive]) == min_consecutive:
            valid_trigger_starts.append(start)

    logging.info("Identified %d valid trigger starts", len(valid_trigger_starts))
    return valid_trigger_starts

# Finds the next trigger start index for an fMRI run given an array of trigger starts
def find_trigger_start(trigger_starts, current_index):
    """
    Finds the index of the start of the trigger signal for the next fMRI run.

    Parameters:
    - trigger_starts (numpy array): Array containing all trigger start indices.
    - current_index (int): The index from which to start searching for the next trigger start.

    Returns:
    - int or None: The index of the next trigger start if found, otherwise None.

    This function searches for the next trigger start after a given index. It uses numpy's searchsorted 
    method to find the appropriate insertion point for the current_index in the trigger_starts array, which 
    corresponds to the next trigger start. If no more triggers are found, it returns None.

    Usage Example:
    next_trigger = find_trigger_start(trigger_starts_array, current_trigger_index)

    Dependencies:
    - numpy for array operations.
    - logging module for logging information.
    """

    logging.info(f"Finding trigger start after index {current_index}.")

    # Find the next trigger start index after the current_index
    next_trigger_index = np.searchsorted(trigger_starts, current_index, side='right')
    
    # If the next_trigger_index is within the bounds of the trigger_starts array
    if next_trigger_index < len(trigger_starts):
        next_trigger_start = trigger_starts[next_trigger_index]
        logging.info(f"Next trigger start found at index {next_trigger_start}.")
        return next_trigger_start
    else:
        logging.error(f"No trigger start found after index {current_index}.")
        return None  # Indicate that there are no more triggers to process

# Segments the data into runs based on the metadata for each fMRI run and the identified trigger starts..
def find_runs(data, all_runs_metadata, trigger_starts, sampling_rate):
    """
    Segments the MRI data into runs based on the metadata for each fMRI run and the identified trigger starts.

    Parameters:
    - data (numpy array): The full MRI data set.
    - all_runs_metadata (dict): A dictionary containing the metadata for each run, keyed by run ID.
    - trigger_starts (list): A list of potential trigger start indices.
    - sampling_rate (float): The rate at which data points are sampled (in Hz).

    Returns:
    - list: A list of dictionaries, each representing a valid run with start and end indices and associated metadata.

    This function processes each run based on its repetition time (TR) and number of volumes, 
    and identifies valid segments of the MRI data corresponding to each run. It skips runs with 
    missing metadata and handles cases where the proposed segment is out of the data bounds.

    Usage Example:
    run_segments = find_runs(data_array, run_metadata, trigger_starts, 500)

    Dependencies:
    - numpy for array operations.
    - logging module for logging information and errors.
    """
    
    runs_info = []
    start_from_index = 0  # Start from the first trigger

    for run_id, metadata in all_runs_metadata.items():
        try:
            repetition_time = metadata.get('RepetitionTime')
            num_volumes = metadata.get('NumVolumes')
            if repetition_time is None or num_volumes is None:
                logging.info(f"RepetitionTime or NumVolumes missing in metadata for {run_id}")
                continue

            logging.info(f"Processing {run_id} with {num_volumes} volumes and TR={repetition_time}s")
            samples_per_volume = int(sampling_rate * repetition_time)

            # Start searching for a valid run from the last used trigger index
            for i in range(start_from_index, len(trigger_starts) - num_volumes + 1):
                expected_interval = samples_per_volume * (num_volumes - 1)
                actual_interval = trigger_starts[i + num_volumes - 1] - trigger_starts[i]
                
                if actual_interval <= expected_interval:
                    start_idx = trigger_starts[i]
                    end_idx = start_idx + num_volumes * samples_per_volume
                    if end_idx > data.shape[0]:
                        logging.info(f"Proposed end index {end_idx} for {run_id} is out of bounds.")
                        continue
                    
                    run_info = {
                        'run_id': run_id,
                        'start_index': start_idx,
                        'end_index': end_idx,
                        'metadata': metadata
                    }
                    runs_info.append(run_info)
                    logging.info(f"Valid run found for {run_id}: start at {start_idx}, end at {end_idx}")
                    start_from_index = i + num_volumes  # Update the starting index for the next run
                    break  # Exit loop after finding a valid start index for this run
                else:
                    logging.info(f"{run_id} at index {i} does not match expected interval.")
            
            if start_from_index >= len(trigger_starts) - num_volumes + 1:
                logging.info(f"No valid segments found for {run_id} after index {start_from_index}.")

        except KeyError as e:
            logging.info(f"Metadata key error for {run_id}: {e}")
        except ValueError as e:
            logging.info(f"ValueError for {run_id}: {e}")
        except Exception as e:
            logging.info(f"Unexpected error while finding runs for {run_id}: {e}")

    logging.info(f"Total number of runs found: {len(runs_info)}")
    if not runs_info:
        logging.info("No runs were found. Please check the triggers and metadata.")

    return runs_info

# Segments the runs based on the information provided by find_runs() and writes them to output files. 
def segment_runs(runs_info, output_dir, metadata_dict, labels, subject_id, session_id):
    """
    Segments the data into runs and writes them to output files based on the information provided by find_runs().

    Parameters:
    - runs_info (list): A list of dictionaries containing the information of each run identified by find_runs.
    - output_dir (str): The directory where output files will be written.
    - metadata_dict (dict): Additional metadata for all runs.
    - labels (list of str): The labels of the data channels.
    - subject_id (str): The identifier for the subject.
    - session_id (str): The identifier for the session.

    Returns:
    - list: A list of file paths to the output files that were written.

    Raises:
    - IOError: If there's an issue writing the output files.

    This function processes each run, potentially applying filtering or normalization, 
    and then writes the processed data to an output file in the specified directory.

    Usage Example:
    output_files = segment_runs(run_info, '/output/dir', metadata, channel_labels, 'sub-01', 'ses-1')

    Dependencies:
    - os module for file operations.
    - logging module for logging information and errors.
    """

    output_files = []
    for run in runs_info:
        run_id = run['run_id']
        data = run['data']
        start_index = run['start_index']
        end_index = run['end_index']
        run_metadata = run['run_metadata']
        task_name = run_metadata['TaskName']  # Extract the TaskName from the metadata
        
        logging.info("Segmenting run %s from index %d to %d", run_id, start_index, end_index)

        # Write the processed data to an output file.
        try:
            # Call the existing write_output_files function with the appropriate parameters
            write_output_files(data, run_metadata, metadata_dict, labels, output_dir, subject_id, session_id, run_id)
            output_file_path = os.path.join(output_dir, f"{subject_id}_{session_id}_task-rest_{run_id}_physio.tsv.gz")
            output_files.append(output_file_path)
            logging.info("Output file for run %s written to %s", run_id, output_file_path)
        except IOError as e:
            logging.error("Failed to write output file for run %s: %s", run_id, e, exc_info=True)
            raise

    return output_files

# Create the metadata dictionary for a run based on the available channel information
def create_metadata_dict(run_info, sampling_rate, bids_labels_list, units_dict):
    """
    Creates a metadata dictionary for an fMRI run, incorporating information about data channels and their properties.

    Parameters:
    - run_info (dict): Information about the run, including its identifier and start index.
    - sampling_rate (float): The sampling rate of the physiological data.
    - bids_labels_list (list): A list of BIDS-compliant labels for the data channels.
    - units_dict (dict): A mapping from BIDS-compliant labels to their respective units.

    Returns:
    - dict: A dictionary containing metadata for the run.

    The function initializes the metadata with common information such as run ID, sampling frequency, and start time.
    It then adds channel-specific metadata, such as description, placement, gain, and filtering settings, for each 
    available channel based on the bids_labels_list.

    Usage Example:
    metadata = create_metadata_dict(run_info, 500, ['cardiac', 'respiratory', 'eda'], {'cardiac': 'mV', 'respiratory': 'cm', 'eda': 'μS'})

    Dependencies:
    - logging module for logging information.
    """

    # Initialize the metadata dictionary with common information
    metadata_dict = {
        
        # Common metadata
        "RunID": run_info['run_id'],
        "SamplingFrequency": sampling_rate, # BIDS label
        "SamplingRate": {
            "Value": sampling_rate,  
            "Units": "Hz" # provide units
        },
        "StartTime": run_info['start_index'] / sampling_rate, # BIDS label
        "StartTimeSec": {
            "Value": run_info['start_index'] / sampling_rate, 
            "Description": "Start time of the current run relative to recording onset",
            "Units": "seconds"
        },
        "StartTimeMin": {
            "Value": (run_info['start_index'] / sampling_rate)/60, 
            "Description": "Start time of the current run relative to recording onset",
            "Units": "minutes"
        },
        "Columns": bids_labels_list,
        "Manufacturer": "Biopac",
        "Acquisition Software": "Acqknowledge",
    }

    # Channel-specific metadata
    channel_metadata = {
        "cardiac": {
            "Description": "Continuous ECG measurement",
            "Placement": "Lead 1",
            "Gain": 500,
            "35HzLPN": "off / 150HzLP",
            "HPF": "0.05 Hz",
        },
        "respiratory": {
            "Description": "Continuous measurements by respiration belt",
            "Gain": 10,
            "LPF": "10 Hz",
            "HPF1": "DC",
            "HPF2": "0.05 Hz",
        },
        "eda": {
            "Description": "Continuous EDA measurement",
            "Placement": "Right plantar instep",
            "Gain": 5,
            "LPF": "1.0 Hz",
            "HPF1": "DC",
            "HPF2": "DC",
        },
        "trigger": {
            "Description": "fMRI Volume Marker",
        },
        "ppg": {
            "Description": "Continuous PPG measurement",
            "Placement": "Left index toe",
            "Gain": 10,
            "LPF": "3.0 Hz",
            "HPF1": "0.5 Hz",
            "HPF2": "0.05 Hz",
        }
    }

    # Add channel-specific metadata to the dictionary if the channel is present
    for channel in channel_metadata:
    
    # Assuming 'channel' is a variable in this function that holds the current channel being processed
        if channel in bids_labels_list:
            channel_specific_metadata = channel_metadata[channel]
            
            # Set the 'Units' dynamically based on the units_dict
            channel_specific_metadata['Units'] = units_dict.get(channel, "Unknown")
            metadata_dict[channel] = channel_specific_metadata

    return metadata_dict

# Writes the segmented data to TSV and JSON files according to the BIDS format
def write_output_files(segmented_data, run_metadata, metadata_dict, bids_labels_list, output_dir, subject_id, session_id, run_id):
    """
    Writes the segmented data for an fMRI run to TSV and JSON files following BIDS format.

    Parameters:
    - segmented_data (numpy array): Data segmented for a specific run.
    - run_metadata (dict): Metadata specific to the run.
    - metadata_dict (dict): Additional metadata for the run.
    - bids_labels_list (list of str): BIDS-compliant labels for the data channels.
    - output_dir (str): Directory where the output files will be written.
    - subject_id (str): Identifier for the subject.
    - session_id (str): Identifier for the session.
    - run_id (str): Identifier for the run.

    The function creates TSV and JSON files named according to BIDS naming conventions.
    It writes the segmented data to a compressed TSV file and the combined metadata to a JSON file.

    Usage Example:
    write_output_files(segmented_data, run_meta, meta_dict, labels, '/output/path', 'sub-01', 'ses-1', 'run-01')

    Dependencies:
    - os module for file operations.
    - pandas for data manipulation and writing TSV files.
    - json for writing JSON files.
    - logging module for logging information and errors.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define filenames based on the BIDS format
        tsv_filename = f"{subject_id}_{session_id}_task-rest_{run_id}_physio.tsv.gz"
        json_filename = f"{subject_id}_{session_id}_task-rest_{run_id}_physio.json"
        
        # Prepare the full file paths
        tsv_file_path = os.path.join(output_dir, tsv_filename)
        json_file_path = os.path.join(output_dir, json_filename)

        # Create a DataFrame with the segmented data and correct labels
        df = pd.DataFrame(segmented_data, columns=bids_labels_list)

        # Save the DataFrame to a TSV file with GZip compression
        df.to_csv(tsv_file_path, sep='\t', index=False, compression='gzip')
        logging.info(f"TSV file written: {tsv_file_path}")

        # Merge the run-specific metadata with the additional metadata
        combined_metadata = {**run_metadata, **metadata_dict}

        # Write the combined metadata to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(combined_metadata, json_file, indent=4)
        logging.info(f"JSON file written: {json_file_path}")

    except Exception as e:
        # Log any exceptions that occur during the file writing process
        logging.error(f"Failed to write output files for {run_id}: {e}", exc_info=True)
        raise

    # Log the successful writing of files
    logging.info(f"Output files for {run_id} written successfully to {output_dir}")

# Plots the segmented data for each run and saves the plots to a png file.
def plot_runs(original_data, segmented_data_list, runs_info, bids_labels_list, sampling_rate, plot_file_path, units_dict, cut_off_duration):
    """
    Visualizes and saves plots of segmented data for each fMRI run.

    Parameters:
    - original_data (np.ndarray): The complete original dataset for background visualization.
    - segmented_data_list (list of np.ndarray): Data arrays for each segmented run.
    - runs_info (list): Information about each run, including indices and metadata.
    - bids_labels_list (list of str): BIDS-compliant labels for the data channels.
    - sampling_rate (int): Sampling rate of the data in Hz.
    - plot_file_path (str): Path to save the plot as a PNG file.
    - units_dict (dict): Dictionary mapping labels to their respective units.

    The function creates a plot for each data channel with the original data in the background and 
    overlays the segmented data for each run. It uses different colors for each run for clear distinction.

    Usage Example:
    plot_runs(orig_data, seg_data_list, runs_info, ['cardiac', 'respiratory'], 500, 'output_plot.png', {'cardiac': 'mV', 'respiratory': 'cm'}, cut_off_duration=0)

    - cut_off_duration (int): Duration in minutes to cut off from the start of the plot. Default is 0 (no cut-off).
    
    Dependencies:
    - numpy for array manipulation.
    - matplotlib for creating plots.
    - logging module for logging information and errors.
    """
    logging.info(f"Cut-off duration in plot_runs: {cut_off_duration} minutes")
    try:
        # Define colors for different runs
        colors = ['purple', 'g', 'b', 'black', 'm', 'y', 'k', 'orange', 'pink', 'violet', 'lime', 'indigo', 'r', 'gold', 'grey', 'brown']

        # Calculate the number of samples to cut off
        cut_off_samples = cut_off_duration * 60 * sampling_rate
        logging.info(f"Cutoff duration: {cut_off_duration} minutes")
        logging.info(f"Cutoff samples: {cut_off_samples}")

        # Adjust original data and time axis if a cut-off is applied
        if cut_off_duration > 0 and original_data.shape[0] > cut_off_samples:
            original_data = original_data[cut_off_samples:] # Adjust this line if needed
            time_axis_original = np.arange(original_data.shape[0]) / sampling_rate / 60
        else:
            time_axis_original = np.arange(original_data.shape[0]) / sampling_rate / 60

        # Create figure and subplots
        fig, axes = plt.subplots(nrows=len(bids_labels_list), ncols=1, figsize=(20, 10))

        # Plot background data
        for i, label in enumerate(bids_labels_list):
            unit = units_dict.get(label, 'Unknown unit')
            axes[i].plot(time_axis_original, original_data[:, i], color='grey', alpha=0.5, label='Background' if i == 0 else "")
            axes[i].set_ylabel(f'{label} ({unit})')

        for segment_index, (segment_data, run_info) in enumerate(zip(segmented_data_list, runs_info)):
            # Calculate adjusted indices
            adjusted_start_index = run_info['start_index'] - cut_off_samples
            adjusted_end_index = run_info['end_index'] - cut_off_samples

            # Ensure indices are within valid range
            adjusted_start_index = max(adjusted_start_index, 0)
            adjusted_end_index = min(adjusted_end_index, original_data.shape[0])

            # Check if there is data to plot after adjustment
            if adjusted_start_index < adjusted_end_index:
                # Calculate time axis relative to the original data
                time_axis_segment = np.arange(adjusted_start_index, adjusted_end_index) / sampling_rate / 60
                time_axis_segment = time_axis_segment[adjusted_start_index - run_info['start_index']: adjusted_end_index - run_info['start_index']]
                
                # Plot segment data
                color = colors[segment_index % len(colors)]
                for i, label in enumerate(bids_labels_list):
                    segment_plot_data = segment_data[adjusted_start_index - run_info['start_index']: adjusted_end_index - run_info['start_index'], i]
                    axes[i].plot(time_axis_segment, segment_plot_data, color=color, label=f'Run {run_info["run_id"]}' if i == 0 else "")
            else:
                logging.info(f"Skipping run {run_info['run_id']} due to empty data segment after adjustment.")

        # Set labels and legend
        axes[-1].set_xlabel('Time (min)')
        handles, labels = [], []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        if handles and labels:
            fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), len(labels)))

        # Adjust layout and save the plot
        fig.tight_layout(rect=[0.05, 0.1, 0.95, 0.97])
        plt.savefig(plot_file_path, dpi=600)
        logging.info(f"Plot saved to {plot_file_path}")
        plt.show()

    except Exception as e:
        logging.error("Failed to plot runs: %s", e, exc_info=True)
        raise

# Main function to orchestrate the conversion process
def main(physio_root_dir, bids_root_dir, cut_off_duration=0):
    """
    Main function to orchestrate the conversion of physiological data to BIDS format.

    Parameters:
    - physio_root_dir (str): Directory containing the physiological .mat files.
    - bids_root_dir (str): Root directory of the BIDS dataset where output files will be saved.

    The function performs the following steps:
    1. Load physiological data from .mat files.
    2. Rename channels according to BIDS conventions.
    3. Extract metadata from associated JSON files.
    4. Segment the data into runs based on triggers and metadata.
    5. Write segmented data to output files in BIDS format.
    6. Plot the physiological data for all runs.

    Usage Example:
    main('/path/to/physio_data', '/path/to/bids_dataset')

    - cut_off_duration (int): Duration in minutes to cut off from the start of the plot. Default is 0 (no cut-off).
   
    Dependencies:
    - Requires several helper functions defined in the same script for loading data, renaming channels,
      extracting metadata, finding runs, segmenting data, writing output files, and plotting.
    """
    print(f"Cut-off duration in main: {cut_off_duration} minutes")

    # Extract subject and session IDs from the path.
    subject_id, session_id = extract_subject_session(physio_root_dir)
    
    # Define output directory for the BIDS dataset.
    output_dir = os.path.join(bids_root_dir, subject_id, session_id, 'func')
            
    # Example values for expected MATLAB file sizes in megabytes
    expected_mat_file_size_range_mb = (1500, 2500)

    # Check MATLAB and TSV files before processing
    if not check_files(physio_root_dir, output_dir, expected_mat_file_size_range_mb):
        print(f"Initial file check failed. Exiting script.")
        return # Exit the script if file check fails.
    
    # Define the known sampling rate
    sampling_rate = 5000  # Replace with the actual sampling rate if different

    # Check MATLAB and TSV files before processing
    if not check_files(physio_root_dir, output_dir, expected_mat_file_size_range_mb):
        print(f"Initial file check failed. Exiting script.")
        return # Exit the script if file check fails.
    
    # Setup logging after extracting subject_id and session_id.
    log_file_path = setup_logging(subject_id, session_id, bids_root_dir)
    logging.info("Processing subject: %s, session: %s", subject_id, session_id)

    try:
 
        # Load physiological data from the .mat file. 
        mat_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-rest_physio.mat")
        labels, data, units = load_mat_file(mat_file_path)
        if data is None or not data.size:
            raise ValueError("Data is empty after loading.")
        logging.info("Data loaded successfully with shape: %s", data.shape)

        # Rename channels according to BIDS conventions
        bids_labels_dictionary, bids_labels_list = rename_channels(labels)

        # Create a dictionary mapping from original labels to units
        original_labels_to_units = dict(zip(labels, units))

        # Create a mapping of original labels to their indices in the data array
        original_label_indices = {label: idx for idx, label in enumerate(labels)}

        # Now create the units_dict by using the bids_labels_dictionary to look up the original labels
        units_dict = {
            bids_labels_dictionary[original_label]: unit
            for original_label, unit in original_labels_to_units.items()
            if original_label in bids_labels_dictionary
        }

        # Filter the data array to retain only the columns with BIDS labels
        segmented_data_bids_only = data[:, [original_label_indices[original] for original in bids_labels_dictionary.keys()]]

        # Now, create the units_dict by using the bids_labels_dictionary to look up the original labels
        units_dict = {bids_labels_dictionary[original_label]: unit 
                    for original_label, unit in original_labels_to_units.items() if original_label in bids_labels_dictionary}
        logging.info("Units dictionary: %s", units_dict)
        logging.info("BIDS labels list: %s", bids_labels_list)

        # Process JSON files to extract metadata for each run
        json_file_paths = glob.glob(os.path.join(bids_root_dir, subject_id, session_id, 'func', '*_bold.json'))
        # logging.info("JSON files found: %s", json_file_paths)
        logging.info("Extracting metadata from JSON files...")

        # Assume json_file_paths is a list of paths to your JSON files
        all_runs_metadata = {}
        processed_jsons = set()

        # Sort json_file_paths based on the run ID number extracted from the file name
        sorted_json_file_paths = sorted(
            json_file_paths,
            key=lambda x: int(re.search(r"run-(\d+)_bold\.json$", x).group(1)))

        # logging.info("Sorted JSON file paths: %s", sorted_json_file_paths)
        # logging.info("Extracting metadata from JSON files...")

        # Now use the sorted_json_file_paths instead of the unsorted json_file_paths
        for json_file_path in sorted_json_file_paths:
            run_id, run_metadata = extract_metadata_from_json(json_file_path, processed_jsons)
            if run_id and run_metadata:  # Checks that neither are None
                all_runs_metadata[run_id] = run_metadata

        # Sort the run IDs based on their numeric part
        sorted_run_ids = sorted(all_runs_metadata.keys(), key=lambda x: int(x.split('-')[1]))

        # Create an ordered dictionary that respects the sorted run IDs
        sorted_all_runs_metadata = OrderedDict((run_id, all_runs_metadata[run_id]) for run_id in sorted_run_ids)

        # Sort the expected runs to ensure they are processed in the correct order
        expected_runs = sorted(all_runs_metadata.keys(), key=lambda x: int(x.split('-')[1]))

        # Find the index for the trigger channel using the BIDS labels list
        if 'trigger' in bids_labels_list:
            trigger_channel_index = bids_labels_list.index('trigger')
            mri_trigger_data = data[:, trigger_channel_index]
        else:
            raise ValueError("Trigger channel not found in BIDS labels.")

        # Extract trigger points with the appropriate threshold and min_consecutive
        # Note: Adjust the threshold and min_consecutive based on your specific data
        trigger_starts = extract_trigger_points(mri_trigger_data, threshold=5, min_consecutive=5)
        if len(trigger_starts) == 0:
            raise ValueError("No trigger points found, please check the threshold and min_consecutive parameters.")
        logging.info("Trigger starts: %s", len(trigger_starts)) 

        # Find runs using the extracted trigger points.
        runs_info = find_runs(data, all_runs_metadata, trigger_starts, sampling_rate)
        if len(runs_info) == 0:
            raise ValueError("No runs were found, please check the triggers and metadata.")

        # Verify that the found runs match the expected runs from the JSON metadata.
        if not runs_info:
            raise ValueError("No runs were found. Please check the triggers and metadata.")
        #logging.info("Runs info: %s", runs_info)

        # Verify that the found runs match the expected runs from the JSON metadata
        expected_runs = set(run_info['run_id'] for run_info in runs_info)
        if expected_runs != set(all_runs_metadata.keys()):
            raise ValueError("Mismatch between found runs and expected runs based on JSON metadata.")

        # Create a mapping from run_id to run_info
        run_info_dict = {info['run_id']: info for info in runs_info}

        # Verify that the found runs match the expected runs from the JSON metadata
        if not set(sorted_run_ids) == set(run_info_dict.keys()):
            raise ValueError("Mismatch between found runs and expected runs based on JSON metadata.")

        # Segment runs and write output files for each run, using sorted_run_ids to maintain order
        output_files = []
        for run_id in sorted_run_ids:
            run_info = run_info_dict[run_id]
            #logging.info("Processing run info: %s", run_info)
            logging.info("Run ID: %s", run_id)
            logging.info("Processing %s", run_id)
            start_index, end_index = run_info['start_index'], run_info['end_index']
            logging.info("start_index: %s", start_index)
            logging.info("end_index: %s", end_index)
            segmented_data = data[start_index:end_index]
            logging.info("Segmented data shape: %s", segmented_data.shape)
            
            # Create the metadata dictionary for the current run
            metadata_dict = create_metadata_dict(run_info, sampling_rate, bids_labels_list, units_dict)
            # logging.info("Metadata dictionary for run %s: %s", run_id, metadata_dict)

            # Call the write_output_files function with the correct parameters
            output_files.append(write_output_files(
                segmented_data_bids_only[start_index:end_index],
                sorted_all_runs_metadata[run_id],
                metadata_dict,
                bids_labels_list,
                output_dir,
                subject_id,
                session_id,
                run_id
            ))       

        # Create a list of segmented data for plotting
        segmented_data_list = [segmented_data_bids_only[run_info['start_index']:run_info['end_index']] for run_info in runs_info]
        logging.info("Segmented data list length: %s", len(segmented_data_list))
        logging.info("Segmented data list shape: %s", segmented_data_list[0].shape)

        # Filter the original data array to retain only the columns with BIDS labels
        data_bids_only = data[:, [original_label_indices[original] for original in bids_labels_dictionary.keys()]]

        # Plot physiological data for all runs with the filtered background data
        if segmented_data_list:
            logging.info("Preparing to plot runs.")
            plot_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-rest_all_runs_physio.png")
            plot_runs(data_bids_only, segmented_data_list, runs_info, bids_labels_list, sampling_rate, plot_file_path, units_dict, cut_off_duration)
        else:
            logging.error("No data available to plot.")

    except Exception as e:
        logging.error("An error occurred in the main function: %s", e, exc_info=True)
        raise

# Main function to run the script from the command line
if __name__ == '__main__':
    """
    Command-line execution entry point for the script.

    This block allows the script to be run directly from the command line. It uses argparse to handle
    command-line arguments, specifically the paths to the directories containing the physiological data
    and the BIDS dataset. These arguments are then passed to the main function of the script.

   Usage:

    python BIDS_process_physio_ses_1.py <physio_root_dir> <bids_root_dir> [--cut_off_duration <duration_in_minutes>]

    Where --cut_off_duration is an optional argument to specify the duration in minutes to cut off from the start of the plot.

    """
    # physio_root_dir = '/path/to/physio_dir'
    # bids_root_dir = '/path/to/bids_root_dir'
    # cut_off_duration = 5

    # Set up an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")

    # The first argument is the root directory containing the matlab physio data.
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")

    # The second argument is the root directory of the BIDS dataset.
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    
    # Optional argument to specify cut-off duration in minutes
    parser.add_argument("--cut_off_duration", type=int, default=0, help="Duration in minutes to cut off from the start of the plot.")
    
    # Parse the arguments provided by the user.
    args = parser.parse_args()
    
    # Starting script messages
    print(f"Starting script with provided arguments.")
    print(f"Physiological data directory: {args.physio_root_dir}")
    print(f"BIDS root directory: {args.bids_root_dir}")
    print(f"Cut-off duration received: {args.cut_off_duration} minutes")

    if args.cut_off_duration > 0:
        print(f"Cut-off duration: {args.cut_off_duration} minutes")

    # Call the main function with the parsed arguments.
    try:
        main(args.physio_root_dir, args.bids_root_dir, args.cut_off_duration)
    except Exception as e:
        logging.error("An error occurred during script execution: %s", e, exc_info=True)
        logging.info("Script execution completed with errors.")
    else:
        logging.info("Script executed successfully.")