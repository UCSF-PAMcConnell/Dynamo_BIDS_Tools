"""
BIDS_process_physio_ses_2.py

Description:
This script processes physiological data acquired through Biopac Acqknowledge during Task fMRI sessions into BIDS-compliant files.
It includes functions for loading data, segmenting physio data based on full run and event types, writing metadata, and generating quality control plots.

Usage:
python BIDS_process_physio_ses_2.py <physio_root_directory> <bids_root_dir>
e.g.,
<dataset_root_dir>/sourcedata/sub-01/ses-01/physio/sub-01_ses-01_task-learn_physio.mat # <physio_root_dir>
<dataset_root_dir>/dataset # <bids_root_dir>

Author: PAMcConnell
Created on: 20231111
Last Modified: 20231111

License: MIT License

Dependencies:
- Python 3.12
- numpy, pandas, matplotlib, scipy

Environment Setup:
To set up the required environment, use the provided environment.yml file with Conda. <datalad.yml>

Change Log:
- 20231111: Initial version

"""

import os                                             # Used for interacting with the operating system, like file path operations.
import re                                             # Regular expressions, useful for text matching and manipulation.
import logging                                        # Logging library, for tracking events that happen when running the software.
import argparse                                       # Parser for command-line options, arguments, and sub-commands.
import scipy.io as sio                                # SciPy module for reading and writing MATLAB files.
import numpy as np                                    # NumPy library for numerical operations on arrays and matrices.
import pandas as pd                                   # Pandas library for data manipulation and analysis.
import matplotlib.pyplot as plt                       # Matplotlib's pyplot, a plotting library.
import json                                           # Library for working with JSON data.
import glob                                           # Used for Unix style pathname pattern expansion.
from collections import OrderedDict                   # Dictionary subclass that remembers the insertion order of keys.
import sys                                            # System-specific parameters and functions.
from matplotlib.lines import Line2D                   # Import Line2D from matplotlib to create custom line objects.

## Helper Functions. 

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
        filemode='w' # 'w' mode overwrites existing log file
    )

    # If you also want to log to console.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Logging setup complete. Log file: {log_file_path}")

# Extract the subject and session IDs from the provided physio root directory path.
def extract_subject_session(physio_root_dir):
    """
    Parameters:
    - physio_root_dir (str): The directory path that includes subject and session information. 
                             This path should follow the BIDS convention, containing 'sub-' and 'ses-' prefixes.

    Returns:
    - subject_id (str): The extracted subject ID.
    - session_id (str): The extracted session ID.

    Raises:
    - ValueError: If the subject_id and session_id cannot be extracted from the path.

    This function assumes that the directory path follows the Brain Imaging Data Structure (BIDS) naming convention. 
    It uses regular expressions to find and extract the subject and session IDs from the path.

    Usage Example:
    subject_id, session_id = extract_subject_session('/path/to/data/sub-01/ses-1/physio')

    Note: This function will raise an error if it cannot find a pattern matching the BIDS convention in the path.
    """

    # Normalize the path to remove any trailing slashes for consistency.
    physio_root_dir = os.path.normpath(physio_root_dir)

    # The pattern looks for 'sub-' followed by any characters until a slash, and similar for 'ses-'.
    match = re.search(r'(sub-[^/]+)/(ses-[^/]+)', physio_root_dir)
    if not match:
        raise ValueError("Unable to extract subject_id and session_id from path: %s", physio_root_dir)
    
    subject_id, session_id = match.groups()

    return subject_id, session_id

# Load a MATLAB (.mat) file containing physiological data and extracts labels, data, and units.
def load_mat_file(mat_file_path):
    """
    Parameters:
    - mat_file_path (str): The file path to the .mat file.

    Returns:
    - labels (array): Names of the physiological data channels.
    - data (array): The actual physiological data.
    - units (array): Units for each channel of physiological data.

    Raises:
    - FileNotFoundError: If the .mat file does not exist at the provided path.
    - KeyError: If the .mat file is missing any of the required keys ('labels', 'data', 'units').
    - Exception: For any other issues encountered while loading or processing the .mat file.

    The function first checks if the specified .mat file exists. It then attempts to load the file 
    and verifies that it contains the required keys. If successful, it extracts and returns the labels, 
    data, and units. Comprehensive logging is implemented for troubleshooting.

    Usage Example:
    labels, data, units = load_mat_file('/path/to/physio_data.mat')

    Note: Ensure that the .mat file is structured with 'labels', 'data', and 'units' keys.
    """
    
    # Verify if the specified .mat file exists.
    if not os.path.isfile(mat_file_path):
        logging.error("MAT file does not exist at %s", mat_file_path)
        raise FileNotFoundError("No MAT file found at the specified path: %s", mat_file_path)
    
    try:
        # Attempt to load the .mat file.
        logging.info(f"Loading MAT file from: {mat_file_path}")
        mat_contents = sio.loadmat(mat_file_path)
        
        # Verify that required keys are in the loaded .mat file.
        required_keys = ['labels', 'data', 'units']
        if not all(key in mat_contents for key in required_keys):
            logging.error("MAT file at %s is missing one of the required keys: %s", mat_file_path, required_keys)
            raise KeyError("MAT file at %s is missing required keys", mat_file_path)
        
        # Extract labels, data, and units.
        labels = mat_contents['labels'].flatten()  # Flatten in case it's a 2D array.
        data = mat_contents['data']
        units = mat_contents['units'].flatten()  # Flatten in case it's a 2D array
        logging.info(f"Successfully loaded MAT file from: {mat_file_path}")
              
    except Exception as e:
        # Log the exception and re-raise to handle it upstream.
        logging.error("Failed to load MAT file from %s: %s", mat_file_path, e)
        raise
    
    return labels, data, units

# Rename physiological data channels according to the BIDS conventions.
def rename_channels(labels):
    """
    Parameters:
    - labels (array): Original names of the physiological data channels.

    Returns:
    - bids_labels_dictionary (dict): Mapping from original labels to BIDS-compliant labels.
    - bids_labels_list (list): A list of the renamed, BIDS-compliant labels.

    The function iterates through the provided original labels and renames them based on predefined 
    mapping to meet BIDS standards. Channels not recognized are omitted with a warning. This function
    is essential for ensuring that physiological data aligns with the BIDS format for further processing.

    Usage Example:
    bids_labels_dict, bids_labels_list = rename_channels(['ECG', 'RSP', 'EDA', 'PPG', 'Digital input'])

    Note: The function expects a specific set of channel names. Make sure to update the mapping 
    if different channels or naming conventions are used.
    """

    # Define the mapping from original labels to BIDS labels.
    original_label_mapping = {
        'ECG': 'cardiac',
        'RSP': 'respiratory',
        'EDA': 'eda',
        'Trigger': 'trigger',
        'PPG': 'ppg',  # Only if exists
        # Add other mappings as required
    }

    # Initialize an empty dictionary and list to store the renamed labels.
    bids_labels_dictionary = {}
    bids_labels_list = []

    # Initialize dictionary and list for storing BIDS-compliant labels.
    for label in labels:
        # Skip any labels for digital inputs
        if 'Digital input' in label:
            continue
        
        # Check and rename the label if it matches one of the keys in original_label_mapping.
        for original, bids in original_label_mapping.items():
            if original in label:
                bids_labels_dictionary[label] = bids
                bids_labels_list.append(bids)
                break
        else:
            logging.warning("Label '%s' does not match any BIDS convention and will be omitted.", label)
    
    # Log the results of the renaming process.
    # logging.info(f"BIDS Labels Dictionary: {bids_labels_dictionary}")
    # logging.info(f"BIDS Labels List: {bids_labels_list}")
    
    return bids_labels_dictionary, bids_labels_list

#  Extracts metadata from a BIDS JSON file and identifies the associated run.
def extract_metadata_from_json(json_file_path, processed_jsons):
    """
    Parameters:
    - json_file_path (str): Path to the JSON file containing metadata for a specific fMRI run.
    - processed_jsons (set): A set that tracks already processed JSON files to avoid duplication.

    Returns:
    - tuple: (run_id, run_metadata), where:
        - run_id (str): The identifier of the fMRI run, extracted from the file name.
        - run_metadata (dict): A dictionary containing extracted metadata.

    Raises:
    - FileNotFoundError: If the specified JSON file does not exist.
    - ValueError: If the run_id cannot be determined or required metadata fields are missing.
    - json.JSONDecodeError: If the JSON file contains invalid JSON.

    This function is critical for parsing and organizing metadata necessary for subsequent data processing steps.
    It verifies the existence of essential fields and logs detailed information for debugging and audit purposes.
    
    Usage Example:
    run_id, metadata = extract_metadata_from_json('/path/to/run-01_bold.json', processed_jsons_set)
    """
   
    # Log the start of metadata extraction.
    logging.info(f"Extracting metadata from JSON file: {json_file_path}")

    # Avoid reprocessing the same file.
    if json_file_path in processed_jsons:
        logging.info("JSON file %s has already been processed.", json_file_path)
        return None, None  # No new metadata to return.
    
    # Check if the JSON file exists.
    if not os.path.isfile(json_file_path):
        logging.error("JSON file does not exist at %s", json_file_path)
        raise FileNotFoundError(f"No JSON file found at the specified path: {json_file_path}")
    
    # Load the JSON file content.
    try:
        with open(json_file_path, 'r') as file:
            metadata = json.load(file)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        raise

    # Extract run_id from the file name.
    run_id_match = re.search(r'run-\d+', json_file_path)
    if not run_id_match:
        raise ValueError(f"Run identifier not found in JSON file name: {json_file_path}")
    run_id = run_id_match.group()

    # Verify essential metadata fields are present.
    required_fields = ['TaskName', 'RepetitionTime', 'NumVolumes']
    run_metadata = {field: metadata.get(field) for field in required_fields}
    if not all(run_metadata.values()):
        missing_fields = [key for key, value in run_metadata.items() if value is None]
        raise ValueError(f"JSON file {json_file_path} is missing required fields: {missing_fields}")

    # Add the JSON file to the set of processed files.
    processed_jsons.add(json_file_path)

    # Return the run ID and extracted metadata.
    run_metadata = {field: metadata[field] for field in required_fields}
    # logging.info(f"Extracted metadata for run {run_id}: {run_metadata}")
    return run_id, run_metadata

# Extracts the starting points of triggers in MRI data based on a specified threshold and minimum consecutive points.
def extract_trigger_points(data, threshold, min_consecutive):
    """
    Parameters:
    - data (numpy.ndarray): The data array from the MRI trigger channel.
    - threshold (float): The threshold value above which a data point is considered part of a trigger.
    - min_consecutive (int): The minimum number of consecutive data points above the threshold required to confirm a trigger start.

    Returns:
    - list: A list of indices in the data array where valid trigger starts are detected.

    This function is integral to preprocessing fMRI data, where accurate identification of trigger points is essential for aligning physiological data with MRI volumes. It uses a simple thresholding approach combined with a criterion for a minimum number of consecutive points to reduce false positives.

    Usage Example:
    trigger_starts = extract_trigger_points(mri_trigger_data, threshold=5, min_consecutive=5)

    Note:
    - This function assumes the trigger channel data is a 1D numpy array.
    - The choice of threshold and min_consecutive parameters may vary based on the specifics of the MRI acquisition and should be validated for each dataset.
    """
    
    # Convert the data points to binary values based on the threshold.
    triggers = (data > threshold).astype(int)
    logging.info(f"Trigger data converted to binary based on threshold {threshold}")

    # Calculate the difference to identify rising edges of the trigger signal.
    diff_triggers = np.diff(triggers, prepend=0)
    potential_trigger_starts = np.where(diff_triggers == 1)[0]

    # Validate trigger starts based on the minimum consecutive points criterion.
    valid_trigger_starts = []
    for start in potential_trigger_starts:
        if np.sum(triggers[start:start+min_consecutive]) == min_consecutive:
            valid_trigger_starts.append(start)

    return valid_trigger_starts

# Finds the index of the next MRI trigger start after a given index in the sequence of trigger starts.
def find_trigger_start(trigger_starts, current_index):
    """
    Parameters:
    - trigger_starts (numpy.ndarray): An array of indices where MRI trigger starts are detected.
    - current_index (int): The current index in the physiological data, used as a reference to find the next trigger start.

    Returns:
    - int or None: The index of the next trigger start if found; otherwise, None.

    This function is essential for processing fMRI physiological data, where identifying the start of each scan run is crucial for accurate data analysis. It works by searching for the nearest index in the trigger_starts array that comes after the current_index.

    Usage Example:
    next_trigger = find_trigger_start(trigger_starts, current_index)

    Note:
    - Ensure trigger_starts is a sorted numpy array.
    - If no trigger start is found after the current_index, the function returns None, indicating the end of the sequence or missing data.
    """
    
    # Finding the appropriate index in the trigger_starts array.
    next_trigger_index = np.searchsorted(trigger_starts, current_index, side='right')
    
    # Verify if the next trigger index is within the bounds of the trigger_starts array.
    if next_trigger_index < len(trigger_starts):
        next_trigger_start = trigger_starts[next_trigger_index]
        logging.info(f"Next trigger start found at index {next_trigger_start}.")
        return next_trigger_start
    else:
        logging.error(f"No trigger start found after index {current_index}.")
        return None  # Indicate that there are no more triggers to process.

# Segments physiological data into individual fMRI runs based on metadata and trigger starts.
def find_runs(data, all_runs_metadata, trigger_starts, sampling_rate):
    """
    Parameters:
    - data (numpy.ndarray): The complete physiological data set.
    - all_runs_metadata (dict): Metadata for each run, keyed by run identifiers.
    - trigger_starts (list): Indices marking the start of each potential fMRI run.
    - sampling_rate (int): The frequency at which data points were sampled.

    Returns:
    - list of dicts: Each dictionary contains the start and end indices of a run and its metadata.

    This function is critical for neuroimaging research where analyses are often conducted on data aligned with individual fMRI runs. It ensures that each run is accurately captured based on the timing information provided in the metadata.

    Usage Example:
    runs_info = find_runs(data, all_runs_metadata, trigger_starts, sampling_rate)

    Note:
    - Ensure the data array and all_runs_metadata are correctly formatted and complete.
    - The function logs detailed information about the process, facilitating troubleshooting and verification.
    """
    runs_info = []
    start_from_index = 0  # Start from the first trigger.

    for run_id, metadata in all_runs_metadata.items():
        try:
            repetition_time = metadata.get('RepetitionTime')
            num_volumes = metadata.get('NumVolumes')
            if repetition_time is None or num_volumes is None:
                logging.info(f"RepetitionTime or NumVolumes missing in metadata for {run_id}")
                continue

            logging.info(f"Searching for a valid {run_id} with {num_volumes} volumes and TR={repetition_time}s")
            samples_per_volume = int(sampling_rate * repetition_time)

            # Finding valid start and end indices for each run.
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
                    logging.info(f"Valid run found for {run_id}: start index at {start_idx} samples, end index at {end_idx} samples")
                    start_from_index = i + num_volumes  # Update the starting index for the next run.
                    break  # Exit loop after finding a valid start index for this run.
                else:
                    logging.info(f"{run_id} at index {i} does not match expected run length, skipping over...")
            
            if start_from_index >= len(trigger_starts) - num_volumes + 1:
                logging.warning(f"No more valid run length matches for {run_id} after index {start_from_index}.")

        except Exception as e:
            logging.error(f"Error processing {run_id}: {e}")

    logging.info(f"Total valid runs identified: {len(runs_info)}")
    if not runs_info:
        logging.warning("No valid runs found. Check triggers and metadata accuracy.")

    return runs_info

# Segments the runs based on the information provided by find_runs() and writes them to output files. 
def segment_runs(runs_info, output_dir, metadata_dict, labels, subject_id, session_id):
    """
    Parameters:
    - runs_info (list): A list of dictionaries, each containing information for a segmented run.
    - output_dir (str): Directory where output files will be saved.
    - metadata_dict (dict): Additional metadata for all runs.
    - labels (list of str): Labels of the data channels.
    - subject_id (str): Identifier for the subject.
    - session_id (str): Identifier for the session.

    Returns:
    - list: A list of file paths to the written output files.

    This function plays a crucial role in organizing and saving segmented physiological data into individual files, 
    each corresponding to a specific fMRI run. It's a vital step for further data analysis in neuroimaging studies.

    Usage Example:
    output_files = segment_runs(runs_info, output_dir, metadata_dict, labels, subject_id, session_id)

    Note:
    - Ensure that the output directory exists and is writable.
    - The function assumes that the data processing method (e.g., filtering, normalization) is defined elsewhere.
    """
    
    output_files = []
    for run in runs_info:
        run_id = run['run_id']
        data = run['data']
        start_index = run['start_index']
        end_index = run['end_index']
        run_metadata = run['run_metadata']
        task_name = run_metadata['TaskName']  # Extract the TaskName from the metadata.
        
        logging.info("Segmenting full run %s from index %d to %d", run_id, start_index, end_index)

        # Write the processed data to an output file.
        try:
            # Call the existing write_output_files function with the appropriate parameters.
            output_file_path = os.path.join(output_dir, f"{subject_id}_{session_id}_task-learn_{run_id}_physio.tsv.gz")
            write_output_files(data, run_metadata, metadata_dict, labels, output_dir, subject_id, session_id, run_id)
            
            output_files.append(output_file_path)
            #logging.info("Output file for run %s written to %s", run_id, output_file_path)

        except IOError as e:
            logging.error("Failed to write output file for full run %s: %s", run_id, e, exc_info=True)
            raise # Propagate the exception for further handling.
    
    logging.info(f"Completed writing output files for all segmented full runs.")
    return output_files

# Creates a metadata dictionary for a run with channel-specific information.
def create_metadata_dict(run_info, sampling_rate, bids_labels_list, units_dict):
    """
    Parameters:
    - run_info (dict): Information about the run, including start index and run ID.
    - sampling_rate (int): The sampling rate of the physiological data.
    - bids_labels_list (list): List of BIDS-compliant labels for data channels.
    - units_dict (dict): Mapping from BIDS labels to their units.

    Returns:
    - dict: A dictionary containing metadata for the run.

    This function is vital for associating physiological data with its metadata, ensuring the data's integrity 
    and facilitating its use in research and analysis.

    Note:
    - The function assumes that the run_info dictionary and units_dict are correctly formatted and provided.
    - Channel-specific metadata is predefined and may need updates based on new information or different setups.
    
    Usage Example:
    run_metadata = create_metadata_dict(run_info, sampling_rate, bids_labels_list, units_dict)
    """

    # Initialize the metadata dictionary with common information.
    metadata_dict = {
        "RunID": run_info['run_id'],
        "SamplingFrequency": sampling_rate,
        "SamplingRate": {
            "Value": sampling_rate,  
            "Units": "Hz"
        },
        "StartTime": run_info['start_index'] / sampling_rate, 
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

    # Channel-specific metadata.
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

    # Add channel-specific metadata to the dictionary if the channel is present.
    for channel in channel_metadata:
    # Assuming 'channel' is a variable in this function that holds the current channel being processed.
        if channel in bids_labels_list:
            channel_specific_metadata = channel_metadata[channel]
            # Set the 'Units' dynamically based on the units_dict
            channel_specific_metadata['Units'] = units_dict.get(channel, "Unknown")
            metadata_dict[channel] = channel_specific_metadata
    
    logging.info(f"Metadata dictionary created for run {run_info['run_id']}")
    return metadata_dict

# Creates metadata for an event segment, including run details and specific event information.
def create_event_metadata_dict(run_info, segment_length, sampling_rate, repetition_time, 
                               bids_labels_list, units_dict, trial_type, 
                               segment_start_time, event_onset, segment_duration):
    """
    Parameters:
    - run_info (dict): Information about the run, including the start index and run ID.
    - segment_length (int): Length of the segment in data points.
    - sampling_rate (int): Sampling rate of the data in Hz.
    - repetition_time (float): Repetition time for the fMRI volumes in seconds.
    - bids_labels_list (list): List of BIDS-compliant labels for data channels.
    - units_dict (dict): Mapping from BIDS labels to their units.
    - trial_type (str): Type of trial, e.g., 'sequence', 'random'.
    - segment_start_time (float): Start time of the segment in seconds.
    - event_onset (float): Onset time of the event relative to the start of the run.
    - segment_duration (float): Duration of the segment in seconds.

    Returns:
    - dict: A dictionary containing metadata for the event segment.

    The function meticulously details each aspect of an event segment, crucial for accurate data analysis 
    and subsequent research applications. It assumes the input parameters are correctly formatted and provided.

    Note:
    - This function can be adapted to different experimental setups and data structures.
    - The channel-specific metadata may require updates to reflect specific measurement techniques or equipment used.
    
    Usage Example:
    event_metadata = create_event_metadata_dict(run_info, segment_length, sampling_rate, repetition_time, 
                                                bids_labels_list, units_dict, trial_type, 
                                                segment_start_time, event_onset, segment_duration)
    """

    # Calculate the number of volumes and the duration of the segment.
    num_volumes_events = int(segment_length / (sampling_rate * repetition_time))
    segment_duration_min = (segment_length / sampling_rate) / 60
    
    # Logging for debugging and verification.
    logging.info(f"NumVolumes: {num_volumes_events}")
    logging.info(f"Segment length: {segment_length} data points")
    logging.info(f"Segment duration: {segment_duration_min} minutes")
    #logging.info((run_info['start_index'] / sampling_rate) + event_onset)

    # Initialize the metadata dictionary with segment-specific information.
    metadata_dict_events = {
        "RunID": run_info['run_id'],
        "NumVolumes": num_volumes_events,
        "RepetitionTime": repetition_time,
        "SamplingFrequency": sampling_rate,
        "StartTime": (run_info['start_index'] / sampling_rate) + event_onset,
        "StartTimeSec": {
            "Value": (run_info['start_index'] / sampling_rate) + event_onset,
            "Description": "Start time of the segment relative to recording onset",
            "Units": "seconds"
        },
        "StartTimeMin": {
            "Value": ((run_info['start_index'] / sampling_rate) + event_onset) / 60,
            "Description": "Start time of the segment relative to recording onset",
            "Units": "minutes"
        },
        "DurationMin": {
            "Value": segment_duration_min,
            "Description": "Duration of the segment",
            "Units": "minutes"
        },
        "TrialType": trial_type,  # Include trial type in the metadata.
        "Columns": bids_labels_list
    }
  
    # Channel-specific metadata.
    channel_event_metadata = {
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

    # Add channel-specific metadata to the dictionary if the channel is present.
    for channel in channel_event_metadata:
    # Assuming 'channel' is a variable in this function that holds the current channel being processed.
        if channel in bids_labels_list:
            channel_specific_event_metadata = channel_event_metadata[channel]
            # Set the 'Units' dynamically based on the units_dict.
            channel_specific_event_metadata['Units'] = units_dict.get(channel, "Unknown")
            metadata_dict_events[channel] = channel_specific_event_metadata

    logging.info(f"Event metadata dictionary created for run {run_info['run_id']} and event type {trial_type}")
    return metadata_dict_events

# Writes segmented data to TSV and JSON files in the Brain Imaging Data Structure (BIDS) format.
def write_output_files(segmented_data, run_metadata, metadata_dict, bids_labels_list, output_dir, subject_id, session_id, run_id):
    """
    Parameters:
    - segmented_data (numpy.ndarray): The data segmented for a run.
    - run_metadata (dict): The metadata for the run.
    - metadata_dict (dict): The additional metadata for the run.
    - bids_labels_list (list of str): The BIDS-compliant labels of the data channels.
    - output_dir (str): The directory to which the files should be written.
    - subject_id (str): The identifier for the subject.
    - session_id (str): The identifier for the session.
    - run_id (str): The identifier for the run.

    Raises:
    - Exception: If any error occurs during the file writing process, it logs the error and re-raises the exception.

    Notes:
    - This function requires the 'pandas' and 'os' libraries for data handling and file operations.
    - The function also utilizes 'json' for JSON file operations and 'logging' for logging errors and information.
    - The output consists of two files: a TSV (Tab Separated Values) file with GZip compression and a JSON file containing metadata.
    - The TSV file name is formatted as '[subject_id]_[session_id]_task-learn_[run_id]_physio.tsv.gz'.
    - The JSON file name is formatted as '[subject_id]_[session_id]_task-learn_[run_id]_physio.json'.
    - This function is designed to work with data adhering to the BIDS format, commonly used in neuroimaging studies.

    Example Usage:
    write_output_files(segmented_data_array, run_metadata_dict, additional_metadata_dict, channel_labels, '/output/path', 'sub-01', 'sess-01', 'run-01')
    """
    try:
        # Ensure the output directory exists.
        os.makedirs(output_dir, exist_ok=True)

        # Define filenames based on the BIDS format.
        tsv_filename = f"{subject_id}_{session_id}_task-learn_{run_id}_physio.tsv.gz"
        json_filename = f"{subject_id}_{session_id}_task-learn_{run_id}_physio.json"
        
        # Prepare the full file paths.
        tsv_file_path = os.path.join(output_dir, tsv_filename)
        json_file_path = os.path.join(output_dir, json_filename)

        # Create a DataFrame with the segmented data and correct labels.
        df = pd.DataFrame(segmented_data, columns=bids_labels_list)

        # Save the DataFrame to a TSV file with GZip compression.
        df.to_csv(tsv_file_path, sep='\t', index=False, compression='gzip')

        # Merge the run-specific metadata with the additional metadata.
        combined_metadata = {**run_metadata, **metadata_dict}

        # Write the combined metadata to a JSON file.
        with open(json_file_path, 'w') as json_file:
            json.dump(combined_metadata, json_file, indent=4)

        # Log the successful writing of files.
        logging.info(f"Run output files for {run_id} written successfully to {output_dir}")
    
    except Exception as e:
        # Log any exceptions that occur during the file writing process.
        # logging.error(f"Failed to write output files for {run_id}: {e}", exc_info=True)
        # Re-raise the exception for further handling
        raise

# Writes output files (.tsv and .json)for each event segment of physiological data in BIDS format.
def write_event_output_files(event_segments, run_metadata, metadata_dict_events, bids_labels_list, output_dir, subject_id, session_id, run_id):
    """
    Parameters:
    - event_segments (list of tuples): Each tuple contains a segment of data (numpy array) and its corresponding trial type (str), start time (float), and duration (float).
    - run_metadata (dict): Metadata for the run.
    - metadata_dict_events (dict): Additional metadata for the events.
    - bids_labels_list (list of str): BIDS-compliant labels of the data channels.
    - output_dir (str): Directory to write the files.
    - subject_id (str): Identifier for the subject.
    - session_id (str): Identifier for the session.
    - run_id (str): Identifier for the run.

    Raises:
    - Exception: If an error occurs during file writing.

    Notes:
    - This function requires 'pandas' for DataFrame operations and 'os', 'json' for file handling.
    - It assumes that event_segments contains tuples with exactly four elements: data, trial type, start time, and duration.
    - The function creates TSV and JSON files for each event segment, naming them according to the BIDS format and including relevant metadata.
    - Error handling is included to manage unexpected segment formats or file writing issues.

    Example Usage:
    write_event_output_files(event_segments, run_metadata, metadata_dict_events, channel_labels, '/output/path', 'sub-01', 'sess-01', 'run-01')
    """
    try:
        # Ensure the output directory exists.
        os.makedirs(output_dir, exist_ok=True)
        #logging.info(f"Output directory '{output_dir}' created successfully.")

        # Loop through each segment and write to individual files.
        for i, segment_info in enumerate(event_segments):

            # Adjust the segment index for logging and naming (starting from 1 instead of 0).
            segment_index = i + 1

            # Unpack the segment_info tuple.
            if len(segment_info) == 4:  # Assuming there are four elements in the tuple.
                segment_data, trial_type, segment_start_time, segment_duration = segment_info
            else:
                logging.error(f"Unexpected event segment format for {run_id}: {segment_info}")
                continue  # Skip this segment.

            # Define filenames based on the BIDS format.
            tsv_event_filename = f"{subject_id}_{session_id}_task-learn_{run_id}_recording-{trial_type}_physio.tsv.gz"
            json_event_filename = f"{subject_id}_{session_id}_task-learn_{run_id}_recording-{trial_type}_physio.json"

            # Prepare file paths.
            tsv_event_file_path = os.path.join(output_dir, tsv_event_filename)
            json_event_file_path = os.path.join(output_dir, json_event_filename)
            # logging.info(f"File paths set for segment {i}: TSV - {tsv_event_file_path}, JSON - {json_event_file_path}")

            # Check if segment data is not empty.
            if segment_data.size == 0:
                logging.warning(f"No data found for event segment {segment_index} of trial type '{trial_type}' in run '{run_id}'. Skipping file writing.")
                continue

            # Create a DataFrame and save to a TSV file.
            df = pd.DataFrame(segment_data, columns=bids_labels_list)
            df.to_csv(tsv_event_file_path, sep='\t', index=False, compression='gzip')
            #logging.info(f"TSV file written for segment {i}, trial type '{trial_type}'.")

            # Write the event metadata to a JSON file.
            with open(json_event_file_path, 'w') as json_file:
                json.dump(metadata_dict_events, json_file, indent=4)
                #logging.info(f"JSON file written for segment {i}, trial type '{trial_type}'.")

            # Log the successful writing of files.
            logging.info(f"Event segment output files for {run_id}, segment {segment_index}, trial type '{trial_type}' written successfully to {output_dir}.")

    except Exception as e:
        # Log any exceptions during the file writing process.
        logging.error(f"Failed to write event output files for {run_id}: {e}", exc_info=True)
        raise

# Plots the segmented data for each run and saves the plots to a PNG file.
def plot_runs(original_data, segmented_data_list, runs_info, bids_labels_list, sampling_rate, plot_file_path, units_dict):
    """
    Parameters:
    - original_data (np.ndarray): The entire original data to be used as a background.
    - segmented_data_list (list of np.ndarray): Each element is an array of data for a run.
    - runs_info (list): Information about each run, including start and end indices and metadata.
    - bids_labels_list (list of str): BIDS-compliant channel labels.
    - sampling_rate (int): The rate at which data was sampled.
    - plot_file_path (str): The file path to save the plot.
    - units_dict (dict): A dictionary mapping labels to their units.

    Raises:
    - Exception: If an error occurs during plotting.

    Notes:
    - This function requires 'numpy' for numerical operations and 'matplotlib.pyplot' for plotting.
    - It creates a multi-panel plot, each panel representing a channel of the data.
    - The original data is plotted as a grey background, and segmented data are overlaid with different colors.
    - The function handles the generation of a dynamic legend and ensures proper layout adjustment.
    - Error handling includes logging the error and re-raising the exception for further handling.

    Example Usage:
    plot_runs(original_data_array, segmented_data_list, runs_info_list, channel_labels, 250, '/path/to/save/plot.png', units_dictionary)
    """

    try:
        # Define a list of colors for different runs.
        colors = [
            'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange',
            'pink', 'purple', 'lime', 'indigo', 'violet', 'gold', 'grey', 'brown'
        ]

        # Create a figure and a set of subplots.
        fig, axes = plt.subplots(nrows=len(bids_labels_list), ncols=1, figsize=(20, 10))

        # Time axis for the original data.
        time_axis_original = np.arange(original_data.shape[0]) / sampling_rate / 60

        # Plot the entire original data as background.
        for i, label in enumerate(bids_labels_list):
            unit = units_dict.get(label, 'Unknown unit')
            axes[i].plot(time_axis_original, original_data[:, i], color='grey', alpha=0.5, label='Background' if i == 0 else "")
            axes[i].set_ylabel(f'{label}\n({unit})')

        # Overlay each segmented run on the background.
        for segment_index, (segment_data, run_info) in enumerate(zip(segmented_data_list, runs_info)):

            # Define time_axis_segment for each segmented run.
            time_axis_segment = np.arange(run_info['start_index'], run_info['end_index']) / sampling_rate / 60
            color = colors[segment_index % len(colors)]  # Cycle through colors.
            for i, label in enumerate(bids_labels_list):
                axes[i].plot(time_axis_segment, segment_data[:, i], color=color, label=f'{run_info["run_id"]}' if i == 0 else "")

        # Set the x-axis label for the bottom subplot.
        axes[-1].set_xlabel('Time (min)')

        # Collect handles and labels for the legend from all axes.
        handles, labels = [], []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()

            # Add the handle/label if it's not already in the list.
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)

        # Only create a legend if there are items to display.
        if handles and labels:
            ncol = min(len(handles), len(labels))
            fig.legend(handles, labels, loc='lower center', ncol=ncol)
        else:
            logging.info("No legend items to display.")

        # Apply tight layout with padding to make room for the legend and axis labels.
        fig.tight_layout(rect=[0.05, 0.1, 0.95, 0.97])  # Adjust the left and bottom values as needed.

        # Save and show the figure.
        plt.savefig(plot_file_path, dpi=600)
        plt.show()  # Comment this line if you want to bypass plot display.
        logging.info(f"Plot saved to {plot_file_path}")

    except Exception as e:
        logging.error("Failed to plot runs: %s", e, exc_info=True)
        raise

# Segments the physiological data based on events described in the events DataFrame, considering 'sequence' and 'random' blocks.
def segment_data_by_events(data, events_df, sampling_rate, run_info, run_id):
    """
    Parameters:
    - data (np.ndarray): The original data array.
    - events_df (pd.DataFrame): Contains 'onset', 'duration', and 'trial_type' for each event.
    - sampling_rate (int): The rate at which data was sampled.
    - run_info (dict): Information about the current run, including the start index.
    - run_id (str): The identifier of the current run.

    Returns:
    - List of tuples: Each tuple contains a segment of data, its trial type, start time, and duration.

    Raises:
    - Exception: If any error occurs during the segmentation process.

    Notes:
    - Special handling is included for runs with only 'random' blocks (e.g., 'run-00', 'run-07').
    - Each segment is represented as a tuple with the structured data, trial type, start time, and duration.
    - This function requires 'numpy' for data handling and 'pandas' for DataFrame operations.
    - Detailed logging is included to track the segmentation process for each type of block.

    Example Usage:
    segments = segment_data_by_events(data_array, events_dataframe, 250, run_info_dict, 'run-01')
    """
    segments = []
    logging.info(f"Starting data segmentation by events for {run_id}.")

    # Special handling for 'random' block only runs (e.g., 'run-00', 'run-07').
    if run_id in ["run-00", "run-07"]:
        
        # Determine end time of the last 'random' event, configure logging outputs.
        last_random_event = events_df[events_df['trial_type'] == 'random'].iloc[-1]
        segment_end_time = int(last_random_event['onset'] + last_random_event['duration']) # seconds relative to the start of the run
        end_index = int((run_info['start_index'] + segment_end_time) * sampling_rate) # samples
        block_end_time_fixed = int(segment_end_time + (run_info['start_index'] / sampling_rate)) # seconds relative to the start of the session
        block_start_time_fixed = int(run_info['start_index'] / sampling_rate) # seconds relative to the start of the session
        block_duration_fixed = int(block_end_time_fixed - block_start_time_fixed) # seconds relative to the start of the session
        fixed_start_index = int(run_info['start_index']) # samples
        fixed_end_index = int(run_info['start_index'] + (segment_end_time * sampling_rate)) # samples
        fixed_index_length = fixed_end_index - fixed_start_index # samples

        # Special handling for 'random' block only runs (e.g., 'run-00', 'run-07') - process logging. 
        logging.info(f"Event fixed random segment Start Time: {block_start_time_fixed} seconds")
        logging.info(f"Event fixed random segment End Time: {block_end_time_fixed} seconds")
        logging.info(f"Event fixed random segment Duration: {block_duration_fixed} seconds")
        logging.info(f"Event fixed random segment Start index: {run_info['start_index']} samples")
        logging.info(f"Event fixed random segment End index: {fixed_end_index} samples")
        logging.info(f"Event fixed random segment length: {fixed_index_length} samples")
        logging.info(
            f"Segment for this fixed random block starts at {block_start_time_fixed} seconds relative to the session start, " 
            f"ends at {block_end_time_fixed} seconds, encompassing a total duration of {block_duration_fixed} seconds " 
            f"and covering {fixed_index_length} data points, from start index {run_info['start_index']} to end index {fixed_end_index} in the data array.")

        # Extract segment for 'random' block.
        segment_data = data[fixed_start_index:fixed_end_index]
        segments.append((segment_data, 'random', run_info['start_index'] / sampling_rate, segment_end_time))
        logging.info(f"Processed 'fixed random' block for {run_id} with random segment end time: {segment_end_time} seconds relative to the start of the run.")
        return segments

    # For other runs, handle both 'sequence' and 'random' events.
    segment_counter = {'sequence': 0, 'random': 0}  # Initialize segment counter as a dictionary

    for trial_type in ['sequence', 'random']:
        block_events = events_df[events_df['trial_type'] == trial_type]
        if block_events.empty:
            #logging.warning(f"No events found for trial type '{trial_type}' in {run_id}.")
            continue

        # Increment the segment counter for this trial type
        segment_counter[trial_type] += 1
        trial_index = segment_counter[trial_type]  # This will be either 1 or 2

        # Calculate start and end times of the block.
        block_start_onset = block_events.iloc[0]['onset']
        last_event = block_events.iloc[-1]
        block_end_time = last_event['onset'] + last_event['duration']
        
        # Log start and end times of the block.
        logging.info(f"'{trial_type}' segment {trial_index} start onset: {block_start_onset} seconds relative to start of the run.")
        logging.info(f"'{trial_type}' segment {trial_index} last onset: {last_event} seconds relative to start of the run.")
        logging.info(f"'{trial_type}' segment {trial_index} end time: {block_end_time} seconds relative to start of the session.")
        
        # Convert times to indices
        start_index = int(block_start_onset * sampling_rate) + run_info['start_index']
        end_index = int(block_end_time * sampling_rate) + run_info['start_index']
        segment_data = data[start_index:end_index]
        index_length = end_index - start_index
        
        # Log start and end indices of the block.
        logging.info(f";{trial_type}' segment {trial_index} start index: {start_index} samples")
        logging.info(f"'{trial_type}' segment {trial_index} end index: {end_index}")
        logging.info(f"'{trial_type}' segment {trial_index} index length: {index_length} samples")
        
        # Log start and end times of the segment.
        segment_start_time = start_index / sampling_rate  # Convert to seconds
        segment_end_time = end_index / sampling_rate  # Convert to seconds
        segment_duration = segment_end_time - segment_start_time

        logging.info(f"'{trial_type}' segment {trial_index} start time: {segment_start_time} seconds realtive to start of the session.")
        logging.info(f"'{trial_type} segment {trial_index} end time: {segment_end_time}")
        logging.info(f"'{trial_type}' segment {trial_index} duration: {segment_duration} seconds")
        
        # Extract data for the segment.
        segment_data = data[start_index:end_index]

        # Append segment info.
        segments.append((segment_data, trial_type, segment_start_time, segment_duration))
        logging.info(f"Segmented '{trial_type}' block {trial_index} for {run_id}: Start time {segment_start_time} s, Duration {segment_duration} s")
        logging.info(
                    f"Segment for this '{trial_type}' starts at {segment_start_time} seconds relative to the session start," 
                    f"ends at {block_end_time} seconds, encompassing a total duration of {segment_duration} seconds " 
                    f"and covering {index_length} data points, from start index {start_index} to end index {end_index} in the data array.")
        
       #segment_counter += 1  # Increment segment counter

    logging.info(f"Data segmentation completed for {run_id}. Total number of segments: {len(segments)}")
    return segments

# Plots segmented data over the original data for visual comparison.
def plot_runs_with_events(original_data, event_segments_by_run, events_df, sampling_rate, plot_events_file_path, units_dict, bids_labels_list, run_info_dict):
    """
    Plots segmented data for each run, overlaying it on the background of the original data. 
    This helps in visual comparison between the original and segmented data.

    Parameters:
    - original_data (np.ndarray): The original full dataset.
    - event_segments_by_run (dict): Segments of data for each run, keyed by run ID.
    - events_df (pd.DataFrame): DataFrame containing event information.
    - sampling_rate (int): Data sampling rate.
    - plot_events_file_path (str): Path to save the plot.
    - units_dict (dict): Dictionary mapping data channels to their units.
    - bids_labels_list (list of str): List of BIDS-compliant labels for data channels.
    - run_info_dict (dict): Dictionary containing run-specific information.

    Notes:
    - The function creates a plot with multiple subplots, each representing a data channel.
    - Original data is plotted as a grey background for reference.
    - Segmented data from different runs are overlaid with distinct colors for 'sequence' and 'random' trial types.
    - This function uses 'matplotlib.pyplot' for plotting and 'numpy' for numerical operations.
    - Logging is used to track the progress and status of the plotting process.

    Raises:
    - Exception: If any error occurs during the plotting process.

    Example Usage:
    plot_runs_with_events(original_data, event_segments_by_run, events_df, 250, '/path/to/save/plot.png', units_dict, channel_labels, run_info)
    """

    # Define colors for different trial types.
    colors = {'sequence': 'red', 'random': 'blue'}

    # Create custom legend handles.
    custom_handles = [Line2D([0], [0], color=colors['sequence'], lw=2, label='Sequence'),
                    Line2D([0], [0], color=colors['random'], lw=2, label='Random')]
    
    # Create figure and subplots.
    fig, axes = plt.subplots(nrows=len(bids_labels_list), ncols=1, figsize=(20, 10))
    #logging.info("Created figure and subplots for event plotting.")

    # Plot the original data as a background reference.
    time_axis_original = np.arange(original_data.shape[0]) / sampling_rate / 60  # Convert to minutes.
    for i, label in enumerate(bids_labels_list):
        axes[i].plot(time_axis_original, original_data[:, i], color='grey', alpha=0.5, label='Background')
        axes[i].set_ylabel(f'{label} ({units_dict.get(label, "Unknown unit")})')

    # Overlay segmented data on top of the original data.
    for run_id, segments in event_segments_by_run.items():
        #logging.info(f"Processing run ID for event plotting: {run_id}")
        for segment_data, trial_type, segment_start_time, segment_duration in segments:
            # Calculate the time axis for the segment.
            start_time = segment_start_time / 60  # Convert to minutes.
            end_time = start_time + segment_duration / 60
            time_axis_segment = np.linspace(start_time, end_time, len(segment_data))

            # Plot each segment.
            for i, label in enumerate(bids_labels_list):
                axes[i].plot(time_axis_segment, segment_data[:, i], color=colors[trial_type])
            #logging.info(f"Plotted {trial_type} segment for {run_id}")

    # Set x-axis label and legend.
    axes[-1].set_xlabel('Time (min)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles))

    # Adjust layout to fit all elements.
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Adjust to accommodate the legend.

    # Save and display the plot.
    plt.savefig(plot_events_file_path, dpi=600)
    plt.show()
    logging.info(f"Plot saved to {plot_events_file_path}")
    
# Main function to orchestrate the conversion of physiological data to BIDS format.
def main(physio_root_dir, bids_root_dir):
    """
    Parameters:
    - physio_root_dir (str): Root directory of the physiological data.
    - bids_root_dir (str): Root directory of the BIDS dataset.

    Notes:
    - This function integrates multiple steps: data loading, channel renaming, metadata processing, 
      event segmentation, file writing, and plotting.
    - It employs a variety of custom functions (e.g., load_mat_file, rename_channels) that should be defined elsewhere.
    - The function is designed to handle exceptions and log detailed information for debugging.
    - Logging is set up to track the process flow and any issues encountered.
    - The sampling rate is hardcoded but can be replaced with a dynamic value if necessary.
    - The function assumes a specific directory structure and file naming convention as per BIDS standards.

    Raises:
    - Exception: If any error occurs during the processing pipeline.
    """

    # Define the known sampling rate.
    sampling_rate = 5000  # Replace with the actual sampling rate if different.

    try:
        # Extract subject and session IDs from the path.
        subject_id, session_id = extract_subject_session(physio_root_dir)
        
        # Setup logging after extracting subject_id and session_id.
        setup_logging(subject_id, session_id, bids_root_dir) # Uncomment this line and helper function to enable archived logging.
        logging.info("Processing subject: %s, session: %s", subject_id, session_id)

        # Load physiological data.
        mat_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-learn_physio.mat")
        labels, data, units = load_mat_file(mat_file_path)
        if data is None or not data.size:
            raise ValueError("Data is empty after loading.")

        # Rename channels according to BIDS conventions.
        bids_labels_dictionary, bids_labels_list = rename_channels(labels)

        # Create a dictionary mapping from original labels to units.
        original_labels_to_units = dict(zip(labels, units))

        # Now create the units_dict by using the bids_labels_dictionary to look up the original labels.
        units_dict = {
            bids_labels_dictionary[original_label]: unit
            for original_label, unit in original_labels_to_units.items()
            if original_label in bids_labels_dictionary
        }
        
        # Create a mapping of original labels to their indices in the data array.
        original_label_indices = {label: idx for idx, label in enumerate(labels)}

        # Filter the data array to retain only the columns with BIDS labels.
        segmented_data_bids_only = data[:, [original_label_indices[original] for original in bids_labels_dictionary.keys()]]

        # Process JSON files to extract metadata for each run.
        json_file_paths = glob.glob(os.path.join(bids_root_dir, subject_id, session_id, 'func', '*_bold.json'))
        logging.info("Extracting run metadata from JSON files...")

        # Assume json_file_paths is a list of paths to your JSON files.
        all_runs_metadata = {}
        processed_jsons = set()

        # Sort json_file_paths based on the run ID number extracted from the file name.
        sorted_json_file_paths = sorted(
            json_file_paths,
            key=lambda x: int(re.search(r"run-(\d+)_bold\.json$", x).group(1)))
        #logging.info("JSON files found: %s", sorted_json_file_paths)

        # Now use the sorted_json_file_paths instead of the unsorted json_file_paths.
        for json_file_path in sorted_json_file_paths:
            run_id, run_metadata = extract_metadata_from_json(json_file_path, processed_jsons)
            if run_id and run_metadata:  # Checks that neither are None.
                all_runs_metadata[run_id] = run_metadata

        # Sort the run IDs based on their numeric part.
        sorted_run_ids = sorted(all_runs_metadata.keys(), key=lambda x: int(x.split('-')[1]))

        # Create an ordered dictionary that respects the sorted run IDs.
        sorted_all_runs_metadata = OrderedDict((run_id, all_runs_metadata[run_id]) for run_id in sorted_run_ids)

        # Sort the expected runs to ensure they are processed in the correct order.
        expected_runs = sorted(all_runs_metadata.keys(), key=lambda x: int(x.split('-')[1]))

        # Find the index for the trigger channel using the BIDS labels list.
        if 'trigger' in bids_labels_list:
            trigger_channel_index = bids_labels_list.index('trigger')
            mri_trigger_data = data[:, trigger_channel_index]
        else:
            raise ValueError("Trigger channel not found in BIDS labels.")

        # Extract trigger points with the appropriate threshold and min_consecutive.
        trigger_starts = extract_trigger_points(mri_trigger_data, threshold=5, min_consecutive=5) # Note: Adjust the threshold and min_consecutive based on your specific data.
        if len(trigger_starts) == 0:
            raise ValueError("No trigger points found, please check the threshold and min_consecutive parameters.")
        logging.info("Trigger starts: %s", len(trigger_starts)) # trigger_starts.

        # Find runs using the extracted trigger points.
        runs_info = find_runs(data, all_runs_metadata, trigger_starts, sampling_rate)
        if len(runs_info) == 0:
            raise ValueError("No runs were found, please check the triggers and metadata.")

        if not runs_info:
            raise ValueError("No runs were found. Please check the triggers and metadata.")
        # logging.info("Runs info: %s", runs_info)

        # Verify that the found runs match the expected runs from the JSON metadata.
        expected_runs = set(run_info['run_id'] for run_info in runs_info)
        if expected_runs != set(all_runs_metadata.keys()):
            raise ValueError("Mismatch between found runs and expected runs based on JSON metadata.")

        # Create a mapping from run_id to run_info.
        run_info_dict = {info['run_id']: info for info in runs_info}

        # Verify that the found runs match the expected runs from the JSON metadata.
        if not set(sorted_run_ids) == set(run_info_dict.keys()):
            raise ValueError("Mismatch between found runs and expected runs based on JSON metadata.")

        event_segments_by_run = {}  # Dictionary to store event segments for each run.

        # Segment runs and write output files for each run, using sorted_run_ids to maintain order.
        output_files = []
        for run_id in sorted_run_ids:
            run_info = run_info_dict[run_id]
            #logging.info("Processing %s", run_id)
            repetition_time = all_runs_metadata[run_id]['RepetitionTime']  # Retrieve repetition time from metadata.
            
            # Construct the events file path.
            events_file_path = os.path.join(
                bids_root_dir, 
                subject_id, 
                session_id, 
                'func', 
                f"{subject_id}_{session_id}_task-learn_{run_id}_events.tsv"
            )
            logging.info(f"Events file path: {events_file_path}")
            
            # Read the events file into events_df.
            try:
                events_df = pd.read_csv(events_file_path, sep='\t')
            except Exception as e:
                logging.error(f"Error reading events file for {run_id}: {e}", exc_info=True)
                continue  # Skip this run if events file cannot be read.
            
            # Log events DataFrame details for debugging. 
            # logging.info("Events DataFrame for %s:\n%s", run_id, events_df.head())

            # Process each segment within the run.
            event_segments = segment_data_by_events(data, events_df, sampling_rate, run_info, run_id)
            event_segments_by_run[run_id] = event_segments

            start_index, end_index = run_info['start_index'], run_info['end_index']
            segmented_data = data[start_index:end_index]
 
            output_dir = os.path.join(bids_root_dir, subject_id, session_id, 'func')
            
            # Create the metadata dictionary for the current run.
            metadata_dict = create_metadata_dict(run_info, sampling_rate, bids_labels_list, units_dict)
            # logging.info("Metadata dictionary for %s: %s", run_id, metadata_dict)
            
            for segment in event_segments:
                logging.info(f"Processing event segment in {run_id}") 
                if len(segment) != 4:
                    logging.error(f"Invalid segment format for {run_id}: {segment}")
                    continue  # Skip malformed segments.

                segment_data, trial_type, segment_start_time, segment_duration = segment
                logging.info(f"Processing segment for trial type {trial_type} in {run_id}")

                segment_length = len(segment_data)
                #logging.info(f"Segment length: {segment_length}")

                # Retrieve the onset time for the segment from events_df.
                event_onset = events_df[events_df['trial_type'] == trial_type]['onset'].iloc[0]
                logging.info(f"Event onset in seconds: {event_onset}")

                # Create event metadata for each segment.
                metadata_dict_events = create_event_metadata_dict(
                    run_info, segment_length, sampling_rate, repetition_time, 
                    bids_labels_list, units_dict, trial_type, 
                    segment_start_time, event_onset, segment_duration
                )

                # logging.info(f"Metadata event dictionary for {run_id}: {metadata_dict_events}")
                logging.info(f"Writing event output files for {run_id}")
                write_event_output_files(
                    [segment], sorted_all_runs_metadata[run_id], metadata_dict_events, 
                    bids_labels_list, output_dir, subject_id, session_id, run_id
                )
                logging.info("Event output files successfully written for run %s, segment of trial type '%s'", run_id, trial_type)

            # Call the write_output_files function with the correct parameters.
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

        # Create a list of segmented data for plotting.
        segmented_data_list = [segmented_data_bids_only[run_info['start_index']:run_info['end_index']] for run_info in runs_info]
        logging.info("Number of runs segmented: %s", len(segmented_data_list))

        # Filter the original data array to retain only the columns with BIDS labels.
        data_bids_only = data[:, [original_label_indices[original] for original in bids_labels_dictionary.keys()]]

        # Plot physiological data for all runs with the filtered background data.
        if segmented_data_list:
            
            # plot_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-learn_all_runs_physio.png")
            
            # Extract the base name of the script without the .py extension.
            script_name = os.path.basename(__file__).replace('.py', '')

            # Construct the log directory path within 'doc/logs'
            log_dir = os.path.join(os.path.dirname(bids_root_dir), 'doc', 'logs', script_name, subject_id)

            # Create the log directory if it doesn't exist.
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Call the plot_runs function with the correct parameters and save plots to the log directory.
            logging.info("Preparing to plot runs.")
            log_file_path_plot_runs = os.path.join(log_dir, f"{subject_id}_{session_id}_task-learn_all_runs_physio.png")
            plot_file_path = log_file_path_plot_runs
            plot_runs(data_bids_only, segmented_data_list, runs_info, bids_labels_list, sampling_rate, plot_file_path, units_dict)

            logging.info("Preparing to plot event blocks.")
            #plot_events_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-learn_all_blocks_physio.png")
            log_file_path_plot_events = os.path.join(log_dir, f"{subject_id}_{session_id}_task-learn_all_blocks_physio.png")
            plot_events_file_path = log_file_path_plot_events
            plot_runs_with_events(data_bids_only, event_segments_by_run, events_df, sampling_rate, plot_events_file_path, units_dict, bids_labels_list, run_info_dict)
        else:
            logging.error("No data available to plot.")

    except Exception as e:
        logging.error("An error occurred in the main function: %s", e, exc_info=True)
        raise

# Main function to run the script from the command line.
if __name__ == '__main__':
    """
    Entry point for the script when run from the command line.
    Uses argparse to parse command-line arguments and passes them to the main function.
    """

    # Create an argument parser.
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir)
