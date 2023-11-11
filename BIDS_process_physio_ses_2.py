import os
import re
import logging
import argparse
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
import glob
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import sys

# Configure basic debug logging
logging.basicConfig(
    filename='process_physio_ses_2.log',
    filemode='w', # a to append, w to overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# def setup_logging(subject_id, session_id, bids_root_dir):
    
#     # Extract the base name of the script without the .py extension
#     script_name = os.path.basename(__file__).replace('.py', '')

#     # Construct the log directory path
#     log_dir = os.path.join(os.path.dirname(bids_root_dir), 'doc', 'logs', script_name)

#     # Create the log directory if it doesn't exist
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     # Construct the log file name
#     log_file_name = f"{subject_id}_{session_id}_{script_name}.log"
#     log_file_path = os.path.join(log_dir, log_file_name)

#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         filename=log_file_path,
#         filemode='w'
#     )

#     # If you also want to log to console
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     console_handler.setFormatter(formatter)
#     logging.getLogger().addHandler(console_handler)

#     logging.info(f"Logging setup complete. Log file: {log_file_path}")

## Helper functions

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
    logging.info("Subject ID: %s, Session ID: %s", subject_id, session_id)
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
        logging.error("MAT file does not exist at %s", mat_file_path)
        raise FileNotFoundError("No MAT file found at the specified path: %s", mat_file_path)
    
    try:
        # Attempt to load the .mat file
        logging.info("Loading MAT file from %s", mat_file_path)
        mat_contents = sio.loadmat(mat_file_path)
        
        # Verify that required keys are in the loaded .mat file
        required_keys = ['labels', 'data', 'units']
        if not all(key in mat_contents for key in required_keys):
            logging.error("MAT file at %s is missing one of the required keys: %s", mat_file_path, required_keys)
            raise KeyError("MAT file at %s is missing required keys", mat_file_path)
        
        # Extract labels, data, and units
        labels = mat_contents['labels'].flatten()  # Flatten in case it's a 2D array
        data = mat_contents['data']
        logging.info("Data loaded successfully with shape: %s", data.shape)
        logging.info("Type of 'data': %s", type(data))
        units = mat_contents['units'].flatten()  # Flatten in case it's a 2D array
        
        # Log the labels and units for error checking
        logging.info("Labels extracted from MAT file: %s", labels)
        logging.info("Units extracted from MAT file: %s", units)
        logging.info("Successfully loaded MAT file from %s", mat_file_path)
        
    except Exception as e:
        # Log the exception and re-raise to handle it upstream
        logging.error("Failed to load MAT file from %s: %s", mat_file_path, e)
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
            logging.warning("Label '%s' does not match any BIDS convention and will be omitted.", label)

    # Debug log to print the renamed labels in the dictionary and the list
    logging.info("BIDS labels dictionary mapping: %s", bids_labels_dictionary)
    logging.info("BIDS labels list after renaming: %s", bids_labels_list)
    
    return bids_labels_dictionary, bids_labels_list

#  Extracts metadata from a JSON file and the associated run_id.
def extract_metadata_from_json(json_file_path, processed_jsons):
    """
    Parameters:
    - json_file_path: str, path to the .json file
    - processed_jsons: set, a set of already processed JSON files.
    Returns:
    - tuple: (run_id, run_metadata) where:
        - run_id: str, the identifier for the run
        - run_metadata: dict, metadata for the run
    Raises:
    - FileNotFoundError, ValueError, json.JSONDecodeError as appropriate
    """
    logging.info("Extracting metadata from %s", json_file_path)
    
    if json_file_path in processed_jsons:
        logging.info("JSON file %s has already been processed.", json_file_path)
        return None, None  # No new metadata to return

    if not os.path.isfile(json_file_path):
        logging.error("JSON file does not exist at %s", json_file_path)
        raise FileNotFoundError(f"No JSON file found at the specified path: {json_file_path}")

    with open(json_file_path, 'r') as file:
        metadata = json.load(file)

    run_id_match = re.search(r'run-\d+', json_file_path)
    if not run_id_match:
        raise ValueError(f"Run identifier not found in JSON file name: {json_file_path}")
    run_id = run_id_match.group()

    required_fields = ['TaskName', 'RepetitionTime', 'NumVolumes']
    run_metadata = {field: metadata.get(field) for field in required_fields}
    if not all(run_metadata.values()):
        missing_fields = [key for key, value in run_metadata.items() if value is None]
        raise ValueError(f"JSON file {json_file_path} is missing required fields: {missing_fields}")

    processed_jsons.add(json_file_path)
    logging.info(f"Successfully extracted metadata for {run_id}: {run_metadata}")

    return run_id, run_metadata

# Identifies potential starts of triggers based on the threshold and minimum number of consecutive points.
def extract_trigger_points(data, threshold, min_consecutive):
    """
    Identifies potential starts of triggers based on the threshold and minimum number of consecutive points.
    Parameters:
    - data: The MRI trigger channel data as a numpy array.
    - threshold: The value above which the trigger signal is considered to start.
    - min_consecutive: Minimum number of consecutive data points above the threshold to consider as a valid trigger start.
    Returns:
    - A list of potential trigger start indices.
    """
    triggers = (data > threshold).astype(int)
    diff_triggers = np.diff(triggers, prepend=0)
    potential_trigger_starts = np.where(diff_triggers == 1)[0]

    valid_trigger_starts = []
    for start in potential_trigger_starts:
        if np.sum(triggers[start:start+min_consecutive]) == min_consecutive:
            valid_trigger_starts.append(start)

    return valid_trigger_starts

# Finds the next trigger start index for an fMRI run given an array of trigger starts
def find_trigger_start(trigger_starts, current_index):
    """""
    Parameters:
    - trigger_starts: np.array, array containing all trigger start indices.
    - current_index: int, the index to start searching for the next trigger.

    Returns:
    - int, the index of the start of the trigger signal for the next run.
    """
    logging.info(f"Finding trigger start after index {current_index}.")

    # Find the index in the trigger_starts array where the current_index would be inserted
    # to maintain order. This is the next trigger start index we want.
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
    Parameters:
    - data: The full MRI data set as a numpy array.
    - all_runs_metadata: A dictionary containing the metadata for each run.
    - trigger_starts: A list of potential trigger start indices.
    - sampling_rate: The rate at which data points are sampled.

    Returns:
    - A list of dictionaries, each containing the start and end indices of a valid run, along with metadata.
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
    Parameters:
    - runs_info: list, a list of dictionaries containing the information of each run identified by find_runs.
    - output_dir: str, the directory where output files will be written.
    - metadata_dict: dict, additional metadata for all runs.
    - labels: list of str, the labels of the data channels.
    - subject_id: str, the identifier for the subject.
    - session_id: str, the identifier for the session.
    Returns:
    - A list of file paths that were written.
    Raises:
    - IOError: If there's an issue writing the output files.
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

        # Perform any necessary processing on the data segment.
        # This could include filtering, normalization, etc.
        # processed_data = process_data(data)

        # Write the processed data to an output file.
        try:
            # Call the existing write_output_files function with the appropriate parameters
            write_output_files(data, run_metadata, metadata_dict, labels, output_dir, subject_id, session_id, run_id)
            output_file_path = os.path.join(output_dir, f"{subject_id}_{session_id}_task-learn_{run_id}_physio.tsv.gz")
            #output_file_path = os.path.join(output_dir, f"{subject_id}_{session_id}_task-{TaskName}_{run_id}_physio.tsv.gz")
            output_files.append(output_file_path)
            logging.info("Output file for run %s written to %s", run_id, output_file_path)
        except IOError as e:
            logging.error("Failed to write output file for run %s: %s", run_id, e, exc_info=True)
            raise

    return output_files

# Create the metadata dictionary for a run based on the available channel information
def create_metadata_dict(run_info, sampling_rate, bids_labels_list, units_dict):
    """
    Parameters:
    - run_info: dict containing information about the run, including the start index.
    - bids_labels_dictionary: dict mapping original labels to BIDS-compliant labels.
    - bids_labels_list: list of BIDS-compliant labels for the channels.
    - units_dict: dict mapping BIDS-compliant labels to units extracted from the .mat file.
    Returns:
    - A metadata dictionary with relevant information for the run.
    """
    # Initialize the metadata dictionary with common information
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

def create_event_metadata_dict(run_info, segment_length, sampling_rate, repetition_time, 
                               bids_labels_list, units_dict, trial_type, 
                               segment_start_time, event_onset, segment_duration):
    """                        
    Creates a metadata dictionary for an event segment based on the available channel information and segment details.

    Parameters:
    - run_info: dict containing information about the run.
    - segment_length: int, length of the segment in data points.
    - sampling_rate: int, sampling rate in Hz.
    - repetition_time: float, repetition time in seconds.
    - bids_labels_list: list of BIDS-compliant labels for the channels.
    - units_dict: dict mapping BIDS-compliant labels to units.

    Returns:
    - A metadata dictionary with relevant information for the event segment.
    """
    # Calculate the number of volumes and the duration of the segment
    num_volumes_events = int(segment_length / (sampling_rate * repetition_time))
    logging.info(f"NumVolumes: {num_volumes_events}")
    segment_duration_min = (segment_length / sampling_rate) / 60
    logging.info(f"Segment length: {segment_length} data points")
    logging.info(f"Segment duration: {segment_duration_min} minutes")
    logging.info((run_info['start_index'] / sampling_rate) + event_onset)


    # Initialize the metadata dictionary with segment-specific information
    metadata_dict_events = {
        "RunID": run_info['run_id'],
        "NumVolumes": num_volumes_events,
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
        "TrialType": trial_type,  # Include trial type in the metadata
        "Columns": bids_labels_list
    }
  
    # Channel-specific metadata
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

    # Add channel-specific metadata to the dictionary if the channel is present
    for channel in channel_event_metadata:
    # Assuming 'channel' is a variable in this function that holds the current channel being processed
        if channel in bids_labels_list:
            channel_specific_event_metadata = channel_event_metadata[channel]
            # Set the 'Units' dynamically based on the units_dict
            channel_specific_event_metadata['Units'] = units_dict.get(channel, "Unknown")
            metadata_dict_events[channel] = channel_specific_event_metadata

    return metadata_dict_events

# Writes the segmented data to TSV and JSON files according to the BIDS format
def write_output_files(segmented_data, run_metadata, metadata_dict, bids_labels_list, output_dir, subject_id, session_id, run_id):
    """
    Parameters:
    - segmented_data: numpy array, the data segmented for a run.
    - run_metadata: dict, the metadata for the run.
    - metadata_dict: dict, the additional metadata for the run.
    - bids_labels_list: list of str, the BIDS-compliant labels of the data channels.
    - output_dir: str, the directory to which the files should be written.
    - subject_id: str, the identifier for the subject.
    - session_id: str, the identifier for the session.
    - run_id: str, the identifier for the run.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define filenames based on the BIDS format
        tsv_filename = f"{subject_id}_{session_id}_task-learn_{run_id}_physio.tsv.gz"
        json_filename = f"{subject_id}_{session_id}_task-learn_{run_id}_physio.json"
        # Prepare the full file paths
        tsv_file_path = os.path.join(output_dir, tsv_filename)
        json_file_path = os.path.join(output_dir, json_filename)

        # Create a DataFrame with the segmented data and correct labels
        df = pd.DataFrame(segmented_data, columns=bids_labels_list)

        # Save the DataFrame to a TSV file with GZip compression
        df.to_csv(tsv_file_path, sep='\t', index=False, compression='gzip')

        # Merge the run-specific metadata with the additional metadata
        combined_metadata = {**run_metadata, **metadata_dict}

        # Write the combined metadata to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(combined_metadata, json_file, indent=4)

        # Log the successful writing of files
        logging.info(f"Output files for {run_id} written successfully to {output_dir}")
    
    except Exception as e:
        # Log any exceptions that occur during the file writing process
        logging.error(f"Failed to write output files for {run_id}: {e}", exc_info=True)
        raise

# Validates the structure and integrity of runs_info by comparing it against run identifiers extracted from fMRI .json files
def validate_runs_info(runs_info, bids_root_dir, subject_id, session_id):
    """
    Parameters:
    - runs_info: list of dicts, each containing metadata and data for a run.
    - bids_root_dir: str, the root directory of the BIDS dataset.
    - subject_id: str, the subject identifier.
    - session_id: str, the session identifier.
    Raises:
    - ValueError: If there are any issues with the integrity of the runs_info.
    """
    # Log the start of validation
    logging.info("Starting validation of the runs_info structure.")
    
    # Construct the path to the fMRI .json files
    json_pattern = os.path.join(bids_root_dir, 'sub-' + subject_id, 'ses-' + session_id, 'func', '*_bold.json')

    # Extract run identifiers from the .json filenames
    json_filenames = glob.glob(json_pattern)
    logging.info("JSON filenames found: %s", json_filenames)
    
    expected_runs = [re.search('run-(\d+)_bold\.json', os.path.basename(f)).group(1) for f in json_filenames]
    logging.info("Expected run identifiers: %s", expected_runs)
    logging.info("Expected run identifiers length: %s", len(expected_runs))
    logging.info("Expected run identifiers type: %s", type(expected_runs))
    logging.info("Expected run identifiers keys: %s", expected_runs.keys())
    logging.info("Expected run identifiers values: %s", expected_runs.values())
    logging.info("Expected run identifiers items: %s", expected_runs.items())
    
    # Log the expected runs
    logging.info("Expected run identifiers: %s", expected_runs)

    # Check if the number of runs in runs_info matches the number of .json files
    if len(runs_info) != len(expected_runs):
        error_message = ("Mismatch between the number of runs found in runs_info (%d) "
                         "and the number of expected runs (%d) based on JSON files.", len(runs_info), len(expected_runs))
        logging.error(error_message)
        raise ValueError(error_message)

    # Check if every run_id in runs_info is among the expected_runs
    for run_info in runs_info:
        if run_info['run_id'] not in expected_runs:
            error_message = ("Run ID %s found in runs_info is not among "
                             "the expected run identifiers: %s", run_info['run_id'], expected_runs)
            logging.error(error_message)
            raise ValueError(error_message)

        # Perform additional checks on each run_info entry
        # Check for essential keys
        essential_keys = ['run_id', 'data', 'start_index', 'end_index']
        for key in essential_keys:
            if key not in run_info:
                error_message = "Missing essential key '%s' in run_info for run ID %s.", key, run_info['run_id']
                logging.error(error_message)
                raise ValueError(error_message)

        # Validate data shape (assuming data is a numpy array)
        if not isinstance(run_info['data'], np.ndarray):
            error_message = "The 'data' key for run ID %s must be a numpy array.", run_info['run_id']
            logging.error(error_message)
            raise ValueError(error_message)

        # Validate indices are within bounds (assuming data is a numpy array)
        data_length = run_info['data'].shape[0]
        logging.info("Data length: %s", data_length)
        logging.info("Start index: %s", run_info['start_index'])
        logging.info("End index: %s", run_info['end_index'])
        logging.info("Data shape: %s", run_info['data'].shape)
        logging.info("Run ID: %s", run_info['run_id'])
        logging.info("Run_info type: %s", type(run_info))
        logging.info("Run_info keys: %s", run_info.keys())
        logging.info("Run_info values: %s", run_info.values())
        logging.info("Run_info items: %s", run_info.items())
        logging.info("Run_info length: %s", len(run_info))

        if not (0 <= run_info['start_index'] < data_length):
            error_message = ("The 'start_index' for run ID %s is out of bounds "
                             "(0, %d).", run_info['run_id'], data_length)
            logging.error(error_message)
            raise ValueError(error_message)
        if not (0 <= run_info['end_index'] <= data_length):
            error_message = ("The 'end_index' for run ID %s is out of bounds "
                             "(0, %d).", run_info['run_id'], data_length)
            logging.error(error_message)
            raise ValueError(error_message)

    # Log successful validation
    logging.info("Validation of the runs_info structure completed successfully.")

    # If all checks pass, the runs_info structure is validated
    return True

# Writes the event-segmented data to TSV and JSON files according to the BIDS format.
def write_event_output_files(event_segments, run_metadata, metadata_dict_events, bids_labels_list, output_dir, subject_id, session_id, run_id):
    """
    Writes output files for each event segment of physiological data.
    
    Parameters:
    - event_segments: list of tuples, each tuple contains a segment of data (numpy array) and its corresponding trial type (str).
    - run_metadata: dict, metadata for the run.
    - metadata_dict_events: dict, additional metadata for the events.
    - bids_labels_list: list of str, BIDS-compliant labels of the data channels.
    - output_dir: str, directory to write the files.
    - subject_id: str, identifier for the subject.
    - session_id: str, identifier for the session.
    - run_id: str, identifier for the run.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory '{output_dir}' created successfully.")

        # Loop through each segment and write to individual files
        for i, segment_info in enumerate(event_segments):
            # Unpack the segment_info tuple
            if len(segment_info) == 4:  # Assuming there are four elements in the tuple
                segment_data, trial_type, segment_start_time, segment_duration = segment_info
            else:
                logging.error(f"Unexpected segment format for {run_id}: {segment_info}")
                continue  # Skip this segment

            # Define filenames based on the BIDS format
            tsv_event_filename = f"{subject_id}_{session_id}_task-learn_{run_id}_recording-{trial_type}_physio.tsv.gz"
            json_event_filename = f"{subject_id}_{session_id}_task-learn_{run_id}_recording-{trial_type}_physio.json"

            # Prepare file paths
            tsv_event_file_path = os.path.join(output_dir, tsv_event_filename)
            json_event_file_path = os.path.join(output_dir, json_event_filename)
            logging.info(f"File paths set for segment {i}: TSV - {tsv_event_file_path}, JSON - {json_event_file_path}")

            # Check if segment data is not empty
            if segment_data.size == 0:
                logging.warning(f"No data found for segment {i} of trial type '{trial_type}' in run '{run_id}'. Skipping file writing.")
                continue

            # Create a DataFrame and save to a TSV file
            df = pd.DataFrame(segment_data, columns=bids_labels_list)
            df.to_csv(tsv_event_file_path, sep='\t', index=False, compression='gzip')
            logging.info(f"TSV file written for segment {i}, trial type '{trial_type}'.")

            # Write the event metadata to a JSON file
            with open(json_event_file_path, 'w') as json_file:
                json.dump(metadata_dict_events, json_file, indent=4)
                logging.info(f"JSON file written for segment {i}, trial type '{trial_type}'.")

            # Log the successful writing of files
            logging.info(f"Event output files for {run_id}, segment {i}, trial type '{trial_type}' written successfully to {output_dir}.")

    except Exception as e:
        # Log any exceptions during the file writing process
        logging.error(f"Failed to write event output files for {run_id}: {e}", exc_info=True)
        raise

# Plots the segmented data for each run and saves the plots to a png file.
def plot_runs(original_data, segmented_data_list, runs_info, bids_labels_list, sampling_rate, plot_file_path, units_dict):
    """
    Parameters:
    - original_data: np.ndarray, the entire original data to be used as a background.
    - segmented_data_list: list of np.ndarray, each element is an array of data for a run.
    - runs_info: list, information about each run, including start and end indices and metadata.
    - bids_labels_list: list of str, BIDS-compliant channel labels.
    - sampling_rate: int, the rate at which data was sampled.
    - plot_file_path: str, the file path to save the plot.
    - units_dict: dict, a dictionary mapping labels to their units.
    """

    try:
        # Define a list of colors for different runs
        colors = [
            'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange',
            'pink', 'purple', 'lime', 'indigo', 'violet', 'gold', 'grey', 'brown'
        ]

        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=len(bids_labels_list), ncols=1, figsize=(20, 10))

        # Time axis for the original data
        time_axis_original = np.arange(original_data.shape[0]) / sampling_rate / 60

        # Plot the entire original data as background
        for i, label in enumerate(bids_labels_list):
            unit = units_dict.get(label, 'Unknown unit')
            axes[i].plot(time_axis_original, original_data[:, i], color='grey', alpha=0.5, label='Background' if i == 0 else "")
            axes[i].set_ylabel(f'{label}\n({unit})')

        # Overlay each segmented run on the background
        for segment_index, (segment_data, run_info) in enumerate(zip(segmented_data_list, runs_info)):
            # Define time_axis_segment for each segmented run
            time_axis_segment = np.arange(run_info['start_index'], run_info['end_index']) / sampling_rate / 60
            color = colors[segment_index % len(colors)]  # Cycle through colors
            for i, label in enumerate(bids_labels_list):
                axes[i].plot(time_axis_segment, segment_data[:, i], color=color, label=f'{run_info["run_id"]}' if i == 0 else "")

        # Set the x-axis label for the bottom subplot
        axes[-1].set_xlabel('Time (min)')

        # Collect handles and labels for the legend from all axes
        handles, labels = [], []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            # Add the handle/label if it's not already in the list
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)

        # Log the handles and labels
        logging.info(f"Legend handles: {handles}")
        logging.info(f"Legend labels: {labels}")

        # Only create a legend if there are items to display
        if handles and labels:
            ncol = min(len(handles), len(labels))
            fig.legend(handles, labels, loc='lower center', ncol=ncol)
        else:
            logging.info("No legend items to display.")

        # Apply tight layout with padding to make room for the legend
        #fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Apply tight layout with padding to make room for the legend and axis labels
        fig.tight_layout(rect=[0.05, 0.1, 0.95, 0.97])  # Adjust the left and bottom values as needed

        # Adjust subplot parameters to give more space
        #plt.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.95)

        # Save and show the figure
        plt.savefig(plot_file_path, dpi=600)
        plt.show()

    except Exception as e:
        logging.error("Failed to plot runs: %s", e, exc_info=True)
        raise

# Parses the events file to extract relevant information.
def parse_events_file(events_file_path):
    """
    Parameters:
    - events_file_path: str, the file path of the events file.
    Returns:
    - DataFrame containing 'onset', 'duration', and 'trial_type' columns.
    """
    try:
        # Read the events file into a DataFrame
        events_df = pd.read_csv(events_file_path, sep='\t')
        logging.info(f"Events file '{events_file_path}' successfully read.")
        return events_df[['onset', 'duration', 'trial_type']]
    except Exception as e:
        logging.error(f"Error reading events file '{events_file_path}': {e}", exc_info=True)
        raise

# Segments the data based on the events described in the events DataFrame, considering the order of 'sequence' and 'random' blocks.
def segment_data_by_events(data, events_df, sampling_rate, run_info, run_id):
    """
    Segments data based on events in the events DataFrame, considering 'sequence' and 'random' blocks.
    Special handling for runs with only 'random' blocks (e.g., run-00, run-07).

    Parameters:
    - data: np.ndarray, the original data array.
    - events_df: DataFrame, contains 'onset', 'duration', and 'trial_type'.
    - sampling_rate: int, the rate at which data was sampled.
    - run_info: dict, information about the current run including the start index.
    - run_id: str, the identifier of the current run.

    Returns:
    - List of tuples, each containing a segment of data, its trial type, start time, and duration.
    """
    segments = []
    logging.info(f"Starting data segmentation by events for {run_id}.")

    # Handling for run-00 and run-07 which only have 'random' blocks
    if run_id in ["00", "07"]:
        # Determine end time of the last 'random' event
        last_random_event = events_df[events_df['trial_type'] == 'random'].iloc[-1]
        logging.info(f"Last 'random' event: {last_random_event}")
        segment_end_time = last_random_event['onset'] + last_random_event['duration']
        logging.info(f"Segment end time: {segment_end_time}")
        end_index = int((run_info['start_index'] + segment_end_time) * sampling_rate)
        logging.info(f"Start index: {run_info['start_index']}")
        logging.info(f"End index: {end_index}")

        # Extract segment for 'random' block
        segment_data = data[:end_index]
        segments.append((segment_data, 'random', run_info['start_index'] / sampling_rate, segment_end_time))
        logging.info(f"Processed 'random' block for {run_id} with segment end time: {segment_end_time} seconds")
        return segments

    # For other runs, handle both 'sequence' and 'random' events
    for trial_type in ['sequence', 'random']:
        block_events = events_df[events_df['trial_type'] == trial_type]
        if block_events.empty:
            logging.warning(f"No events found for trial type '{trial_type}' in {run_id}.")
            continue

        # Calculate segment start and end times in seconds
        block_start_onset = block_events.iloc[0]['onset']
        logging.info(f"Block start onset: {block_start_onset}")
        segment_start_time = (run_info['start_index'] + (block_start_onset * sampling_rate)) / sampling_rate
        logging.info(f"Segment start time: {segment_start_time}")
        last_event = block_events.iloc[-1]
        logging.info(f"Last event: {last_event}")
        block_end_time = last_event['onset'] + last_event['duration']
        logging.info(f"Block end time: {block_end_time}")
        segment_end_time = (run_info['start_index'] + (block_end_time * sampling_rate)) / sampling_rate
        logging.info(f"Segment end time: {segment_end_time}")

        # Calculate segment duration
        segment_duration = segment_end_time - segment_start_time
        logging.info(f"Segment duration: {segment_duration} seconds")
        
        # Extract data for the segment
        start_index = int(block_start_onset * sampling_rate) + run_info['start_index']
        logging.info(f"Start index: {start_index}")
        end_index = int(block_end_time * sampling_rate) + run_info['start_index']
        logging.info(f"End index: {end_index}")
        segment_data = data[start_index:end_index]

        # Append segment info
        segments.append((segment_data, trial_type, segment_start_time, segment_duration))
        logging.info(f"Segmented '{trial_type}' block for {run_id}: Start time {segment_start_time} s, Duration {segment_duration} s")

    logging.info(f"Data segmentation completed for {run_id}. Total number of segments: {len(segments)}")
    return segments

# Plots segmented data over the original data for visual comparison.
def plot_runs_with_events(original_data, event_segments_by_run, events_df, sampling_rate, plot_events_file_path, units_dict, bids_labels_list, run_info_dict):
    """
    Plots segmented data for each run, overlaying it on the background of the original data.

    Parameters:
    - original_data (np.ndarray): The original full dataset.
    - event_segments_by_run (dict): Segments of data for each run.
    - events_df (DataFrame): DataFrame containing event information.
    - sampling_rate (int): Data sampling rate.
    - plot_events_file_path (str): Path to save the plot.
    - units_dict (dict): Dictionary mapping data channels to their units.
    - bids_labels_list (list): List of BIDS-compliant labels for data channels.
    - run_info_dict (dict): Dictionary containing run-specific information.
    """
    # Define colors for different trial types
    colors = {'sequence': 'red', 'random': 'blue'}

    # Create figure and subplots
    fig, axes = plt.subplots(nrows=len(bids_labels_list), ncols=1, figsize=(20, 10))
    logging.info("Created figure and subplots for event plotting.")

    # Plot the original data as a background reference
    time_axis_original = np.arange(original_data.shape[0]) / sampling_rate / 60  # Convert to minutes
    for i, label in enumerate(bids_labels_list):
        axes[i].plot(time_axis_original, original_data[:, i], color='grey', alpha=0.5, label='Background')
        axes[i].set_ylabel(f'{label} ({units_dict.get(label, "Unknown unit")})')

    # Overlay segmented data on top of the original data
    for run_id, segments in event_segments_by_run.items():
        logging.info(f"Processing run ID for event plotting: {run_id}")
        for segment_data, trial_type, segment_start_time, segment_duration in segments:
            # Calculate the time axis for the segment
            start_time = segment_start_time / 60  # Convert to minutes
            end_time = start_time + segment_duration / 60
            time_axis_segment = np.linspace(start_time, end_time, len(segment_data))

            # Plot each segment
            for i, label in enumerate(bids_labels_list):
                axes[i].plot(time_axis_segment, segment_data[:, i], color=colors[trial_type], label=f'{trial_type.capitalize()} - {run_id}')
            logging.info(f"Plotted {trial_type} segment for {run_id}")

    # Set x-axis label and legend
    axes[-1].set_xlabel('Time (min)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles))

    # Adjust layout to fit all elements
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Adjust to accommodate the legend

    # Save and display the plot
    plt.savefig(plot_events_file_path, dpi=300)
    plt.show()
    logging.info(f"Plot saved to {plot_events_file_path}")

# Main function to orchestrate the conversion process
def main(physio_root_dir, bids_root_dir):
    # Define the known sampling rate
    sampling_rate = 5000  # Replace with the actual sampling rate if different

    try:
        # Extract subject and session IDs from the path
        subject_id, session_id = extract_subject_session(physio_root_dir)
        
        # Setup logging after extracting subject_id and session_id
        #setup_logging(subject_id, session_id, bids_root_dir) # Uncomment this line and helper function to enable archived logging

        logging.info("Processing subject: %s, session: %s", subject_id, session_id)

        # Load physiological data
        mat_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-learn_physio.mat")
        labels, data, units = load_mat_file(mat_file_path)
        if data is None or not data.size:
            raise ValueError("Data is empty after loading.")
        logging.info("Data loaded successfully with shape: %s", data.shape)

        # Rename channels according to BIDS conventions
        bids_labels_dictionary, bids_labels_list = rename_channels(labels)

        # Create a dictionary mapping from original labels to units
        original_labels_to_units = dict(zip(labels, units))

        # Now create the units_dict by using the bids_labels_dictionary to look up the original labels
        units_dict = {
            bids_labels_dictionary[original_label]: unit
            for original_label, unit in original_labels_to_units.items()
            if original_label in bids_labels_dictionary
        }
        logging.info("Units dictionary: %s", units_dict)
        logging.info("BIDS labels list: %s", bids_labels_list)
        
        # Create a mapping of original labels to their indices in the data array
        original_label_indices = {label: idx for idx, label in enumerate(labels)}

        # Filter the data array to retain only the columns with BIDS labels
        segmented_data_bids_only = data[:, [original_label_indices[original] for original in bids_labels_dictionary.keys()]]

        # Process JSON files to extract metadata for each run
        json_file_paths = glob.glob(os.path.join(bids_root_dir, subject_id, session_id, 'func', '*_bold.json'))
        logging.info("JSON files found: %s", json_file_paths)
        logging.info("Extracting metadata from JSON files...")

        # Assume json_file_paths is a list of paths to your JSON files
        all_runs_metadata = {}
        processed_jsons = set()

        # Sort json_file_paths based on the run ID number extracted from the file name
        sorted_json_file_paths = sorted(
            json_file_paths,
            key=lambda x: int(re.search(r"run-(\d+)_bold\.json$", x).group(1)))

        logging.info("Sorted JSON file paths: %s", sorted_json_file_paths)
        logging.info("Extracting metadata from JSON files...")

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
        logging.info("Trigger starts: %s", len(trigger_starts)) # trigger_starts)

        # Find runs using the extracted trigger points
        runs_info = find_runs(data, all_runs_metadata, trigger_starts, sampling_rate)
        if len(runs_info) == 0:
            raise ValueError("No runs were found, please check the triggers and metadata.")

        if not runs_info:
            raise ValueError("No runs were found. Please check the triggers and metadata.")
        logging.info("Runs info: %s", runs_info)

        # Verify that the found runs match the expected runs from the JSON metadata
        expected_runs = set(run_info['run_id'] for run_info in runs_info)
        if expected_runs != set(all_runs_metadata.keys()):
            raise ValueError("Mismatch between found runs and expected runs based on JSON metadata.")

        # Create a mapping from run_id to run_info
        run_info_dict = {info['run_id']: info for info in runs_info}

        # Verify that the found runs match the expected runs from the JSON metadata
        if not set(sorted_run_ids) == set(run_info_dict.keys()):
            raise ValueError("Mismatch between found runs and expected runs based on JSON metadata.")

        event_segments_by_run = {}  # Dictionary to store event segments for each run

        # Segment runs and write output files for each run, using sorted_run_ids to maintain order
        output_files = []
        for run_id in sorted_run_ids:
            run_info = run_info_dict[run_id]
            logging.info("Processing %s", run_id)
            repetition_time = all_runs_metadata[run_id]['RepetitionTime']  # Retrieve repetition time from metadata
            
            # Construct the events file path
            events_file_path = os.path.join(
                bids_root_dir, 
                subject_id, 
                session_id, 
                'func', 
                f"{subject_id}_{session_id}_task-learn_{run_id}_events.tsv"
            )
            logging.info(f" Events file path: {events_file_path}")
            
            # Read the events file into events_df
            try:
                events_df = pd.read_csv(events_file_path, sep='\t')
            except Exception as e:
                logging.error(f"Error reading events file for {run_id}: {e}", exc_info=True)
                continue  # Skip this run if events file cannot be read
            
            # Log events DataFrame details
            logging.info("Events DataFrame for %s:\n%s", run_id, events_df.head())

            # Process each segment within the run
            event_segments = segment_data_by_events(data, events_df, sampling_rate, run_info, run_id)
            event_segments_by_run[run_id] = event_segments

            start_index, end_index = run_info['start_index'], run_info['end_index']
            logging.info("start_index: %s", start_index)
            logging.info("end_index: %s", end_index)
            segmented_data = data[start_index:end_index]
 
            output_dir = os.path.join(bids_root_dir, subject_id, session_id, 'func')
            
            # Create the metadata dictionary for the current run
            metadata_dict = create_metadata_dict(run_info, sampling_rate, bids_labels_list, units_dict)
            logging.info("Metadata dictionary for %s: %s", run_id, metadata_dict)
            
            # Process each segment within the run
            event_segments = segment_data_by_events(data, events_df, sampling_rate, run_info, run_id)
            logging.info("Event segments for %s: %s", run_id, event_segments)
            logging.info(f"Event Segmens Type: {type(event_segments)}")
            event_segments_by_run[run_id] = event_segments
            #logging.info(f"Event segments for {run_id}: {event_segments}")

            for segment in event_segments:
                logging.info(f"Processing segment in {run_id}")
                logging.info(f"Segment type: {type(segment)}")   
                if len(segment) != 4:
                    logging.error(f"Invalid segment format for {run_id}: {segment}")
                    continue  # Skip malformed segments

                segment_data, trial_type, segment_start_time, segment_duration = segment
                logging.info(f"Processing segment for trial type {trial_type} in {run_id}")

                segment_length = len(segment_data)
                logging.info(f"Segment length: {segment_length}")

                # Retrieve the onset time for the segment from events_df
                event_onset = events_df[events_df['trial_type'] == trial_type]['onset'].iloc[0]
                logging.info(f"Event onset: {event_onset}")

                # Create event metadata for each segment
                metadata_dict_events = create_event_metadata_dict(
                    run_info, segment_length, sampling_rate, repetition_time, 
                    bids_labels_list, units_dict, trial_type, 
                    segment_start_time, event_onset, segment_duration
                )

                logging.info(f"Metadata event dictionary for {run_id}: {metadata_dict_events}")
                
                success = write_event_output_files(
                    [segment], sorted_all_runs_metadata[run_id], metadata_dict_events, 
                    bids_labels_list, output_dir, subject_id, session_id, run_id
                )
                if success:
                    logging.info("Output files successfully written for run %s, segment of trial type '%s'", run_id, trial_type)
                else:
                    logging.error("Failed to write output files for run %s, segment of trial type '%s'", run_id, trial_type)

                logging.info(f"Written output files for {run_id}, segment of trial type '{trial_type}'")

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
        #logging.info("SEgmented data list shape: %s", segmented_data_list[0].shape)

        # Filter the original data array to retain only the columns with BIDS labels
        data_bids_only = data[:, [original_label_indices[original] for original in bids_labels_dictionary.keys()]]

        # Plot physiological data for all runs with the filtered background data
        if segmented_data_list:
            logging.info("Preparing to plot runs.")
            plot_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-learn_all_runs_physio.png")
            plot_runs(data_bids_only, segmented_data_list, runs_info, bids_labels_list, sampling_rate, plot_file_path, units_dict)

            logging.info("Preparing to plot event blocks.")
            plot_events_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-learn_all_blocks_physio.png")
            plot_runs_with_events(data_bids_only, event_segments_by_run, events_df, sampling_rate, plot_events_file_path, units_dict, bids_labels_list, run_info_dict)
        else:
            logging.error("No data available to plot.")

    except Exception as e:
        logging.error("An error occurred in the main function: %s", e, exc_info=True)
        raise

# Main function to run the script from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir)
