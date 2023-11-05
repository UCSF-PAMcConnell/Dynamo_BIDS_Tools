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

# Configure logging
logging.basicConfig(
    filename='process_physio_ses_1.log',
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


# Extracts the indices where MRI trigger signals start
def extract_trigger_points(mri_trigger_data, threshold=5):
    """
    Parameters:
    - mri_trigger_data: The MRI trigger channel data as a numpy array.
    - threshold: The value above which the trigger signal is considered to start.
    Returns:
    - A numpy array of indices where triggers start.
    """
    try:
        # Log threshold
        logging.info("Extracting trigger points with threshold: %d", threshold)

        triggers = (mri_trigger_data > threshold).astype(int)
        diff_triggers = np.diff(triggers, prepend=0)
        trigger_starts = np.where(diff_triggers == 1)[0]

        # Log trigger_starts for debugging purposes
        # logging.info(f"Type of trigger_starts from extract_trigger_points(): {type(trigger_starts)}")
        logging.info("Length of trigger_starts from extract_trigger_points(): %d", len(trigger_starts))
         # logging.info(f"Shape of trigger_starts from extract_trigger_points(): {trigger_starts.shape}")

        # Log mri_trigger_data for debugging purposes
        # logging.info(f"Type of mri_trigger_data from extract_trigger_points(): {type(mri_trigger_data)}")
        # logging.info(f"Length of mri_trigger_data from extract_trigger_points(): {len(mri_trigger_data)}")
        # logging.info(f"Shape of mri_trigger_data from extract_trigger_points(): {mri_trigger_data.shape}")

        # Log triggers for debugging purposes
        # logging.info(f"Type of triggers from extract_trigger_points(): {type(triggers)}")
        # logging.info(f"Length of triggers from extract_trigger_points(): {len(triggers)}")
        # logging.info(f"Shape of triggers from extract_trigger_points(): {triggers.shape}")

        # Log diff_triggers for debugging purposes
        # logging.info(f"Type of diff_triggers from extract_trigger_points(): {type(diff_triggers)}")
        # logging.info(f"Length of diff_triggers from extract_trigger_points(): {len(diff_triggers)}")
        # logging.info(f"Shape of diff_triggers from extract_trigger_points(): {diff_triggers.shape}")
        
        return trigger_starts
    
    except ValueError as e:
        logging.error("ValueError in extracting trigger points: %s", e, exc_info=True)
        raise
    except TypeError as e:
        logging.error("TypeError in extracting trigger points: %s", e, exc_info=True)
        raise
    except Exception as e:
        logging.error("Unexpected error in extracting trigger points: %s", e, exc_info=True)
        raise

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

# Finds runs based on the MRI trigger data and metadata 
def find_runs(data, all_runs_metadata, trigger_starts, sampling_rate):
    if data.size == 0:
        raise ValueError("Data array is empty.")
    logging.info("Data shape: %s", data.shape)

    for run_id, metadata in all_runs_metadata.items():
        if 'RepetitionTime' not in metadata or 'NumVolumes' not in metadata:
            raise KeyError("Required metadata keys missing for run %s", run_id)

    logging.info("Finding runs based on MRI trigger data and metadata...")
    logging.info("Starting find_runs...")
    logging.info("Data shape: %s", data.shape)
    logging.info("Sampling rate: %d", sampling_rate)
    logging.info("Number of triggers found: %d", len(trigger_starts))

    runs_info = []
    current_index = 0

    logging.info("Current index: %d", current_index)
    for run_id, metadata in all_runs_metadata.items():
        logging.info("Processing metadata for run_id: %s", run_id)
        logging.info("%s", metadata)
        logging.info("%s", all_runs_metadata)
        try:
            repetition_time = metadata['RepetitionTime']
            num_volumes = metadata['NumVolumes']
            if repetition_time is None or num_volumes is None:
                raise KeyError("RepetitionTime or NumVolumes missing in metadata for run %s", run_id)
            logging.info("RepetitionTime: %s, NumVolumes: %s", repetition_time, num_volumes)

            samples_per_volume = int(sampling_rate * repetition_time)
            logging.info("Samples per volume for run %s: %d", run_id, samples_per_volume)

            start_index = find_trigger_start(trigger_starts, current_index)
            logging.info("Start index found for run %s: %s", run_id, start_index)
            if start_index is None:
                logging.error("No valid start index found for run %s after index %d", run_id, current_index)
                continue

            end_index = start_index + num_volumes * samples_per_volume
            if end_index is None:
                logging.error("No valid end index found for run %s after index %s", run_id, start_index)
                continue
            logging.info("End index calculated for run %s: %d", run_id, end_index)

            if end_index > data.shape[0]:
                logging.error("Run %s extends beyond the length of the data. Check the metadata and triggers.", run_id)
                raise ValueError("End index out of bounds for run %s", run_id)

            run_info = {
                'run_id': run_id,
                'start_index': start_index,
                'end_index': end_index,
                'metadata': metadata
            }
            logging.info("Found run info: %s", run_info)
            runs_info.append(run_info)
            logging.info("Run %s added to runs_info: %s", run_id, runs_info)

            current_index = end_index
            logging.info("Current index updated to %d", current_index)
            logging.info("Run %s found from index %d to %d", run_id, start_index, end_index)

        except KeyError as e:
            logging.error("Metadata key error for run %s: %s", run_id, e)
            raise
        except ValueError as e:
            logging.error("ValueError for run %s: %s", run_id, e)
            raise
        except Exception as e:
            logging.error("Unexpected error while finding runs for run %s: %s", run_id, e, exc_info=True)
            raise
    logging.info("Total number of runs found: %d", len(runs_info))
    logging.info("Runs info: %s", runs_info)
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

        logging.info("Segmenting run %s from index %d to %d", run_id, start_index, end_index)

        # Perform any necessary processing on the data segment.
        # This could include filtering, normalization, etc.
        # processed_data = process_data(data)

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
        "SamplingFrequency": {
            "Value": sampling_rate,  
            "Units": "Hz"
        },
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
        tsv_filename = f"{subject_id}_{session_id}_task-rest_{run_id}_physio.tsv.gz"
        json_filename = f"{subject_id}_{session_id}_task-rest_{run_id}_physio.json"

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
        logging.info(f"Output files for run {run_id} written successfully to {output_dir}")
    
    except Exception as e:
        # Log any exceptions that occur during the file writing process
        logging.error(f"Failed to write output files for run {run_id}: {e}", exc_info=True)
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

# Plots the segmented data for each run and saves the plots to a PDF.
def plot_runs(segmented_data_list, runs_info, bids_labels_list, sampling_rate, plot_file_path):
    """
    Parameters:
    - segmented_data_list: list of np.ndarray, each element is an array of data for a run.
    - runs_info: list, information about each run, including start and end indices and metadata.
    - bids_labels_list: list of str, BIDS-compliant channel labels.
    - sampling_rate: int, the rate at which data was sampled.
    - plot_file_path: str, the file path to save the plot.
    """
    try:
        logging.info("Starting to plot runs.")
        # Check if the data list is empty
        logging.info("Segmented data list: %s", segmented_data_list)
        if not segmented_data_list:
            raise ValueError("Segmented data list is empty, cannot plot runs.")

        with PdfPages(plot_file_path) as pdf:
            for segment_index, (segment_data, run_metadata) in enumerate(zip(segmented_data_list, runs_info)):
                # Check if the data segment is not empty
                if isinstance(segment_data, np.ndarray) and segment_data.size > 0:
                    # Create a figure for the segment
                    fig, axes = plt.subplots(nrows=len(bids_labels_list), ncols=1, figsize=(10, 8))
                    time_axis = np.linspace(0, len(segment_data) / sampling_rate, num=len(segment_data))

                    for i, label in enumerate(bids_labels_list):
                        axes[i].plot(time_axis, segment_data[:, i], label=label)
                        axes[i].set_title('Run %s - %s' % (run_metadata["run_id"], label))
                        axes[i].set_xlabel('Time (s)')
                        axes[i].set_ylabel('Amplitude')
                        axes[i].legend(loc='upper right')

                    plt.tight_layout()
                    pdf.savefig(fig)  # Save the current figure into a pdf page
                    plt.show() # Display the current figure for debugging
                    #plt.close() # uncomment to close the figure to free memory
                else:
                    logging.error("Data segment %d is empty, cannot plot this run.", segment_index)
                    continue  # Skip this segment and continue with the next

        logging.info("All runs have been plotted and saved to %s", plot_file_path)

    except Exception as e:
        logging.error("Failed to plot runs: %s", e, exc_info=True)
        raise  # This should be inside the except block

# Main function to orchestrate the conversion process
def main(physio_root_dir, bids_root_dir):
    # Define the known sampling rate
    sampling_rate = 5000  # Replace with the actual sampling rate if different

    try:
        # Extract subject and session IDs from the path
        subject_id, session_id = extract_subject_session(physio_root_dir)
        logging.info("Processing subject: %s, session: %s", subject_id, session_id)

        # Load physiological data
        mat_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-rest_physio.mat")
        labels, data, units = load_mat_file(mat_file_path)
        if data is None or not data.size:
            raise ValueError("Data is empty after loading.")
        logging.info("Data loaded successfully with shape: %s", data.shape)

        # Rename channels based on BIDS format
        bids_labels_dictionary, bids_labels_list = rename_channels(labels)

        # Create a mapping from the original labels to units, excluding 'Digital input'
        original_labels_to_units = {label: unit for label, unit in zip(labels, units) if label not in bids_labels_dictionary.keys()}

        # Create a mapping of original labels to their indices in the data array
        original_label_indices = {label: idx for idx, label in enumerate(labels)}

        # Filter the data array to retain only the columns with BIDS labels
        segmented_data_bids_only = data[:, [original_label_indices[original] for original in bids_labels_dictionary.keys()]]

        # Now, create the units_dict by using the bids_labels_dictionary to look up the original labels
        units_dict = {bids_labels_dictionary[original_label]: unit 
                    for original_label, unit in original_labels_to_units.items() if original_label in bids_labels_dictionary}

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
        trigger_starts = extract_trigger_points(mri_trigger_data)

        # Now call find_runs() with the sorted metadata
        runs_info = find_runs(data, sorted_all_runs_metadata, trigger_starts, sampling_rate)
        logging.info("Trigger starts: %s", len(trigger_starts)) # trigger_starts)

        if not runs_info:
            raise ValueError("No runs were found. Please check the triggers and metadata.")
        logging.info("Runs info: %s", runs_info)

        # Verify that the found runs match the expected runs from the JSON metadata
        if not set(expected_runs) == set([run_info['run_id'] for run_info in runs_info]):
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
            logging.info("Processing run info: %s", run_info)
            logging.info("Run ID: %s", run_id)
            logging.info("Processing run %s", run_id)
            start_index, end_index = run_info['start_index'], run_info['end_index']
            logging.info("start_index: %s", start_index)
            logging.info("end_index: %s", end_index)
            segmented_data = data[start_index:end_index]
            logging.info("Segmented data shape: %s", segmented_data.shape)
            output_dir = os.path.join(bids_root_dir, subject_id, session_id, 'func')
            
            # Create the metadata dictionary for the current run
            metadata_dict = create_metadata_dict(run_info, sampling_rate, bids_labels_list, units_dict)
            
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
        segmented_data_list = [data[run_info['start_index']:run_info['end_index']] for run_info in runs_info]
        logging.info("Segmented data list: %s", segmented_data_list)

        # Plot physiological data for all runs
        if segmented_data_list:
            logging.info("Preparing to plot runs.")
            plot_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-rest_all_runs_physio.png")
            plot_runs(segmented_data_list, runs_info, bids_labels_list, sampling_rate, plot_file_path)
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
