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
        logging.info(f"Data loaded successfully with shape: {data.shape}")
        logging.info(f"Type of 'data': {type(data)}")
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

# Extracts metadata for each fMRI run from associated JSON file and updates the processed_jsons set.
def extract_metadata_from_json(json_file_path, processed_jsons):
    """
    Parameters:
    - json_file_path: str, path to the .json file
    - processed_jsons: set, a set of paths to already processed JSON files.
      This set is modified in-place to include the current json_file_path if the file is processed successfully.
    Returns:
    - run_metadata: dict, a nested dictionary with keys as run identifiers and values as metadata dictionaries
    Raises:
    - FileNotFoundError: If the JSON file does not exist at the specified path.
    - ValueError: If the JSON file is missing required fields.
    - json.JSONDecodeError: If the JSON file is not properly formatted.
    """
    logging.info(f"Extracting metadata from {json_file_path}")

    # Initialize the dictionary that will hold metadata for all runs
    all_runs_metadata = {}

    # Skip processing if this file has already been processed
    if json_file_path in processed_jsons:
        logging.info(f"JSON file {json_file_path} has already been processed.")
        return all_runs_metadata

    # Check if the file exists
    if not os.path.isfile(json_file_path):
        logging.error(f"JSON file does not exist at {json_file_path}")
        raise FileNotFoundError(f"No JSON file found at the specified path: {json_file_path}")

    try:
        # Attempt to open and read the JSON file
        with open(json_file_path, 'r') as file:
            metadata = json.load(file)

        # Assuming the JSON file name contains the run identifier like 'run-01'
        run_id = re.search(r'run-\d+', json_file_path)
        if not run_id:
            raise ValueError(f"Run identifier not found in JSON file name: {json_file_path}")
        run_id = run_id.group()

        # Extract only the required fields for this run
        run_metadata = {
            'TaskName': metadata.get('TaskName'),
            'RepetitionTime': metadata.get('RepetitionTime'),
            'NumVolumes': metadata.get('NumVolumes')
        }

        # Check if all required fields for this run were found
        if not all(run_metadata.values()):
            missing_fields = [key for key, value in run_metadata.items() if value is None]
            logging.error(f"Missing required metadata fields in {json_file_path}: {missing_fields}")
            raise ValueError(f"JSON file {json_file_path} is missing required fields: {missing_fields}")

        # Store this run's metadata using the run identifier as the key
        all_runs_metadata[run_id] = run_metadata

        # Add this file to the set of processed JSON files
        processed_jsons.add(json_file_path)

        # Uncomment below for debugging purposes to log the extracted metadata
        # logging.info(f"Successfully extracted metadata for {run_id}: {run_metadata}")

    except json.JSONDecodeError as e:
        # Log an error if the JSON file is not properly formatted
        logging.error(f"Error decoding JSON from file {json_file_path}: {e}")
        raise

    return all_runs_metadata

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
        logging.info(f"Extracting trigger points with threshold: {threshold}")

        triggers = (mri_trigger_data > threshold).astype(int)
        diff_triggers = np.diff(triggers, prepend=0)
        trigger_starts = np.where(diff_triggers == 1)[0]

        # Log trigger_starts for debugging purposes
        logging.info(f"Type of trigger_starts from extract_trigger_points(): {type(trigger_starts)}")
        logging.info(f"Length of trigger_starts from extract_trigger_points(): {len(trigger_starts)}")
        logging.info(f"Shape of trigger_starts from extract_trigger_points(): {trigger_starts.shape}")

        #Log mri_trigger_data for debugging purposes
        logging.info(f"Type of mri_trigger_data from extract_trigger_points(): {type(mri_trigger_data)}")
        logging.info(f"Length of mri_trigger_data from extract_trigger_points(): {len(mri_trigger_data)}")
        logging.info(f"Shape of mri_trigger_data from extract_trigger_points(): {mri_trigger_data.shape}")

        # Log triggers for debugging purposes
        logging.info(f"Type of triggers from extract_trigger_points(): {type(triggers)}")
        logging.info(f"Length of triggers from extract_trigger_points(): {len(triggers)}")
        logging.info(f"Shape of triggers from extract_trigger_points(): {triggers.shape}")

        # Log diff_triggers for debugging purposes
        logging.info(f"Type of diff_triggers from extract_trigger_points(): {type(diff_triggers)}")
        logging.info(f"Length of diff_triggers from extract_trigger_points(): {len(diff_triggers)}")
        logging.info(f"Shape of diff_triggers from extract_trigger_points(): {diff_triggers.shape}")
        
        return trigger_starts
    except Exception as e:
        logging.error("Failed to extract trigger points", exc_info=True)
        raise

# Identifies the start and end indices for runs in MRI data based on trigger points and metadata.
def find_runs(data, all_runs_metadata, mri_trigger_data, sampling_rate=5000):
    """
    Parameters:
    - data: numpy array, the MRI data.
    - all_runs_metadata: dict, a dictionary containing metadata for all runs, keyed by run identifier.
    - mri_trigger_data: numpy array, the MRI trigger channel data.
    - sampling_rate: int, the sampling rate of the MRI data.
    Returns:
    - runs_info: list, a list of dictionaries, each containing a run's data and start and end indices.
    Raises:
    - KeyError: If a required metadata key is missing.
    - IndexError: If computed indices are out of data bounds.
    - ValueError: If data integrity checks fail or not enough trigger points for expected runs.
    - Exception: For any other unexpected errors.
    """
    try:
        runs_info = []
        trigger_starts = extract_trigger_points(mri_trigger_data)

        # Log trigger_starts for debugging purposes
        logging.info(f"Type of trigger_starts from find_runs(): {type(trigger_starts)}")
        logging.info(f"Length of trigger_starts from find_runs(): {len(trigger_starts)}")
        logging.info(f"Shape of trigger_starts from find_runs(): {trigger_starts.shape}")

        # Log trigger_starts for debugging (verbose)
        logging.info(f"Trigger points identified from find_runs(): {trigger_starts}")

        for run_id, run_metadata in all_runs_metadata.items():
            repetition_time = run_metadata['RepetitionTime']
            num_volumes = run_metadata['NumVolumes']
            samples_per_volume = int(sampling_rate * repetition_time)

            logging.info(f"Processing {run_id} with Repetition time: {repetition_time} and Number of volumes: {num_volumes}")

            # Verify data integrity: ensure there are enough samples for the expected runs and volumes
            expected_samples = num_volumes * samples_per_volume
            if len(data) < expected_samples:
                raise ValueError(f"The data array does not contain enough samples for the expected number of runs and volumes for {run_id}.")

            # Handle edge case: check if there are enough trigger points for the expected number of runs
            if len(trigger_starts) < num_volumes:
                raise ValueError(f"Not enough trigger points for the expected number of runs for {run_id}.")

            # Find the start index for this run based on the trigger points
            start_idx = trigger_starts[num_volumes - 1]  # Assuming trigger points are ordered
            end_idx = start_idx + num_volumes * samples_per_volume

            # Boundary checks: ensure end_idx does not go beyond the length of data
            if end_idx > len(data):
                logging.warning(f"End index {end_idx} for {run_id} goes beyond the length of the data. Trimming to the data length.")
                end_idx = len(data)

            # Extract the segment of data corresponding to this run
            segment = data[start_idx:end_idx, :]
            runs_info.append({'run_id': run_id, 'data': segment, 'start_index': start_idx, 'end_index': end_idx})

            logging.info(f"{run_id} data segment identified from index {start_idx} to {end_idx}")

        return runs_info
    except KeyError as e:
        logging.error(f"Metadata key error for run: {e}", exc_info=True)
        raise
    except IndexError as e:
        logging.error(f"Indexing error: {e}", exc_info=True)
        raise
    except ValueError as e:
        logging.error(f"Value error: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while identifying runs", exc_info=True)
        raise

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

        logging.info(f"Segmenting run {run_id} from index {start_index} to {end_index}")

        # Perform any necessary processing on the data segment.
        # This could include filtering, normalization, etc.
        # processed_data = process_data(data)

        # Write the processed data to an output file.
        try:
            # Call the existing write_output_files function with the appropriate parameters
            write_output_files(data, run_metadata, metadata_dict, labels, output_dir, subject_id, session_id, run_id)
            output_file_path = os.path.join(output_dir, f"{subject_id}_{session_id}_task-rest_{run_id}_physio.tsv.gz")
            output_files.append(output_file_path)
            logging.info(f"Output file for run {run_id} written to {output_file_path}")
        except IOError as e:
            logging.error(f"Failed to write output file for run {run_id}: {e}", exc_info=True)
            raise

    return output_files

# Create the metadata dictionary for a run based on the available channel information
def create_metadata_dict(run_info, sampling_rate, bids_labels_dictionary, bids_labels_list, units_dict):
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
        if channel in bids_labels_dictionary.values():
            channel_specific_metadata = channel_metadata[channel]
            # Set the 'Units' dynamically based on the units_dict
            channel_specific_metadata['Units'] = units_dict.get(channel, "Unknown")
            metadata_dict[channel] = channel_specific_metadata

    return metadata_dict

# Writes the segmented data to TSV and JSON files according to the BIDS format.
def write_output_files(segmented_data, run_metadata, metadata_dict, labels, output_dir, subject_id, session_id, run_id):
    """   
    Parameters:
    - segmented_data: numpy array, the data segmented for a run.
    - run_metadata: dict, the metadata for the run.
    - metadata_dict: dict, the additional metadata for the run.
    - labels: list of str, the labels of the data channels.
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

        # Extract the BIDS-compliant labels, excluding any 'Digital input'
        bids_labels = [label for label in labels if label != 'Digital input']

        # If 'Digital input' exists, remove the corresponding column from the data
        if 'Digital input' in labels:
            digital_input_index = labels.index('Digital input')
            segmented_data = np.delete(segmented_data, digital_input_index, axis=1)
            bids_labels = [label for i, label in enumerate(bids_labels) if i != digital_input_index]

        # Create a DataFrame with the segmented data and correct labels
        df = pd.DataFrame(segmented_data, columns=bids_labels)

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
        logging.error(f"Failed to write output files for run {run_id}", exc_info=True)
        raise

# Plot the physiological data for all runs and save the figure to a specified path.
def plot_runs(data, runs, bids_labels_list, sampling_rate, original_labels, plot_file_path):
    """
    Parameters:
    - data: list of numpy arrays, where each array is the data for a run.
    - runs: list of dicts, each containing metadata for a run.
    - bids_labels_list: list of str, the labels of the data in BIDS format.
    - sampling_rate: float, the sampling rate of the data.
    - original_labels: list of str, the original labels of the data before BIDS conversion.
    - plot_file_path: str, the file path where the plot will be saved.
    """

    logging.info("Starting to plot runs.")
    try:
        logging.info(f"Type of 'data': {type(data)}")
        logging.info(f"Type of 'runs': {type(runs)}")
        # Check if data and runs lists are not empty and have the same length
        if data.size == 0 or len(runs) == 0:
        #if not data or not runs:
            raise ValueError("Data and runs lists must be non-empty.")
        if len(data) != len(runs):
            raise ValueError("Data and runs lists must have the same length.")

        # Create a mapping from BIDS labels to indices in the original data array
        label_indices = {bids_label: original_labels.index(bids_label)
                         for bids_label in bids_labels_list}

        # Initialize the figure and axes
        num_subplots = len(bids_labels_list)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(15, num_subplots * 2), sharex=True)
        if num_subplots == 1:
            axes = [axes]  # Ensure axes is always a list, even for one subplot

        # Plot each label
        for idx, label in enumerate(bids_labels_list):
            ax = axes[idx]  # Use the subplot index
            for run_idx, run_data in enumerate(data):
                run = runs[run_idx]
                # Log the run metadata to see if it contains useful information
                logging.info(f"Metadata for Run {run_idx+1}: {run}")                       
                run_time = np.arange(run_data.shape[0]) / sampling_rate
                ax.plot(run_time, run_data[:, label_indices[label]], label=f"Run {run_idx+1} {label}")

            ax.legend(loc='upper right')
            ax.set_ylabel(label)

        # Final touches to the plot
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(plot_file_path, dpi=300)  # Save the figure
        plt.show()  # Display the plot
        logging.info(f"Runs plotted and saved successfully to {plot_file_path}")
    except Exception as e:
        logging.error("Failed to plot runs", exc_info=True)
        raise



# Main function to run the script from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir)