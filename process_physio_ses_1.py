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

# Extracts required metadata from the .json file associated with each fMRI run
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

        # Check run_metadata type
        #logging.info(f"Successfully extracted run_metadata (type: {type(run_metadata)}): {run_metadata}")

    except json.JSONDecodeError as e:
        # Log an error if the JSON file is not properly formatted
        logging.error(f"Error decoding JSON from file {json_file_path}: {e}")
        raise
    
    return run_metadata

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
        triggers = (mri_trigger_data > threshold).astype(int)
        diff_triggers = np.diff(triggers, prepend=0)
        trigger_starts = np.where(diff_triggers == 1)[0]
        #logging.info(f"Extracted {len(trigger_starts)} trigger points.")
        return trigger_starts
    except Exception as e:
        logging.error("Failed to extract trigger points", exc_info=True)
        raise

# Identifies runs within the MRI data based on trigger signals and run metadata
def find_runs(data, run_metadata, mri_trigger_data, sampling_rate=5000):
    """
    Parameters:
    - data: The MRI data as a numpy array.
    - run_metadata: A dictionary containing metadata about the run.
    - mri_trigger_data: The MRI trigger channel data as a numpy array.
    - sampling_rate: The sampling rate of the MRI data.
    Returns:
    - A list of dictionaries, each containing a run's data and start index.
    """
    try:
        # Extract run metadata
        repetition_time = run_metadata['RepetitionTime']
        logging.info(f"Repetition time: {repetition_time}")
        num_volumes_per_run = run_metadata['NumVolumes']
        logging.info(f"Number of volumes per run: {num_volumes_per_run}")
        samples_per_volume = int(sampling_rate * repetition_time)
        #logging.info(f"Samples per volume: {samples_per_volume}")
        
        # Extract trigger points from the MRI trigger data
        trigger_starts = extract_trigger_points(mri_trigger_data)
        # logging.info(f"Trigger starts: {trigger_starts}")
        # logging.info(f"Number of trigger starts: {len(trigger_starts)}")
        # logging.info(mri_trigger_data)

        runs = []
        current_run = []
        # Loop through all but the last trigger start to prevent index out of bounds
        for i in range(len(trigger_starts) - 1):
            # Add current trigger to the run if we have not reached the required number of volumes
            if len(current_run) < num_volumes_per_run:
                current_run.append(trigger_starts[i])
            # Check if we have reached a new run or the end of the current run
            if len(current_run) == num_volumes_per_run or trigger_starts[i+1] - trigger_starts[i] > samples_per_volume:
                # If we have a complete run, store it
                if len(current_run) == num_volumes_per_run:
                    start_idx = current_run[0]
                    end_idx = start_idx + num_volumes_per_run * samples_per_volume
                    segment = data[start_idx:end_idx, :]
                    runs.append({'data': segment, 'start_index': start_idx})
                # Reset current run for the next iteration
                current_run = []
        # Check for any remaining triggers that might form a run
        if len(current_run) == num_volumes_per_run:
            start_idx = current_run[0]
            end_idx = start_idx + num_volumes_per_run * samples_per_volume
            segment = data[start_idx:end_idx, :]
            runs.append({'data': segment, 'start_index': start_idx})
        
        return runs
    except Exception as e:
        logging.error("Failed to find runs", exc_info=True)
        raise

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

# Write the segmented data to TSV and JSON files
def write_output_files(segmented_data, run_metadata, metadata_dict, labels, output_dir, subject_id, session_id, run_id):
    """
    Parameters:
    - segmented_data: numpy array, the data segmented for a run.
    - metadata: dict, the metadata for the run.
    - output_dir: str, the directory to which the files should be written.
    - run_id: str, the identifier for the run.
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Construct the filenames
        tsv_filename = f"{subject_id}_{session_id}_task-rest_{run_id}_physio.tsv.gz"
        json_filename = f"{subject_id}_{session_id}_task-rest_{run_id}_physio.json"

        # Full paths
        tsv_file_path = os.path.join(output_dir, tsv_filename)
        json_file_path = os.path.join(output_dir, json_filename)

        # Get the BIDS labels in order, excluding 'Digital input'
        bids_labels_list = [metadata_dict['Columns'][i] for i, label in enumerate(labels) if label != 'Digital input']

        # Get the index of 'Digital input' using numpy
        digital_input_index = np.where(labels == 'Digital input')[0]
        if digital_input_index.size > 0:
            # Remove 'Digital input' from both the labels and the data
            segmented_data = np.delete(segmented_data, digital_input_index, axis=1)
            bids_labels_list = [label for i, label in enumerate(metadata_dict['Columns']) if i not in digital_input_index]

        # Write the TSV file with correct column headers
        pd.DataFrame(segmented_data, columns=bids_labels_list).to_csv(tsv_file_path, sep='\t', index=False, compression='gzip')
        
        # Before writing the JSON file, merge run_metadata with channel metadata
        combined_metadata = {**run_metadata, **metadata_dict}
        with open(json_file_path, 'w') as json_file:
            json.dump(combined_metadata, json_file, indent=4)
        
        logging.info(f"Output files for run {run_id} written successfully to {output_dir}")
    except Exception as e:
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
        # Check if data and runs lists are not empty and have the same length
        if not data or not runs:
            raise ValueError("Data and runs lists must be non-empty.")
        if len(data) != len(runs):
            raise ValueError("Data and runs lists must have the same length.")
        
        # Create a mapping from BIDS labels to indices in the original data array
        label_indices = {bids_label: original_labels.index(original_label)
                         for original_label, bids_label in rename_channels(original_labels)[0].items()
                         if bids_label in bids_labels_list}

        # Initialize the figure and axes
        num_subplots = len(bids_labels_list)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 10), sharex=True)
        if num_subplots == 1:
            axes = [axes]  # Ensure axes is always a list, even for one subplot

        # Plot each label
        for label in bids_labels_list:
            if label not in label_indices:
                logging.error(f"Label '{label}' not found in BIDS labels.")
                continue  # Skip labels not found

            label_index = label_indices[label]
            ax = axes if num_subplots == 1 else axes[label_index]

            # Plot the full data for the current label
            time = np.arange(data.shape[0]) / sampling_rate
            ax.plot(time, data[:, label_index], label=f"Full Data {label}", color='lightgrey')

            # Plot each run for the current label
            for run_idx, run in enumerate(runs):
                start_time_of_run = run['start_index'] / sampling_rate
                run_length = run['data'].shape[0]
                run_time = start_time_of_run + np.arange(run_length) / sampling_rate
                ax.plot(run_time, run['data'][:, label_index], label=f"Run {run_idx+1} {label}")

            ax.legend(loc='upper right')

        # Final touches to the plot
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(plot_file_path, dpi=300)  # Save the figure
        plt.show()  # Display the plot
        logging.info(f"Runs plotted and saved successfully to {plot_file_path}")
    except Exception as e:
        logging.error("Failed to plot runs", exc_info=True)
        raise e

# Main function to orchestrate the conversion process
def main(physio_root_dir, bids_root_dir):
    logging.info("Starting main processing function.")

    try:
        # Extract subject and session IDs from the path
        subject_id, session_id = extract_subject_session(physio_root_dir)
        logging.info(f"Processing subject: {subject_id}, session: {session_id}")

        # Construct the .mat file path
        mat_file_name = f"{subject_id}_{session_id}_task-rest_physio.mat"
        mat_file_path = os.path.join(physio_root_dir, mat_file_name)

        # Load physiological data from the .mat file
        labels, data, units = load_mat_file(mat_file_path)
        logging.info("Physiological data loaded successfully.")

        # Rename channels based on BIDS format and create units dictionary
        bids_labels_dictionary, bids_labels_list = rename_channels(labels)
        # After renaming the channels
        bids_labels_dictionary, bids_labels_list = rename_channels(labels)
        logging.info(f"BIDS Labels: {bids_labels_list}")

        # Confirm 'cardiac' is in the BIDS labels list
        if 'cardiac' not in bids_labels_list:
            logging.error("Expected 'cardiac' label is missing from BIDS labels.")
            # Handle the missing label appropriately
        units_dict = {bids_label: unit for label, unit, bids_label in zip(labels, units, bids_labels_list) if label != 'Digital input'}
        logging.info("Channels renamed according to BIDS format and units dictionary created.")

        # Find the index of the trigger channel
        trigger_original_label = next((orig_label for orig_label, bids_label in bids_labels_dictionary.items() if bids_label == 'trigger'), None)
        if trigger_original_label is None:
            raise ValueError("Trigger label not found in BIDS labels dictionary.")
        trigger_channel_index = labels.tolist().index(trigger_original_label)
        #logging.info(f"Trigger channel index identified: {trigger_channel_index}")

        # Extract trigger channel data
        trigger_channel_data = data[:, trigger_channel_index]
        #logging.info(f"Trigger channel data extracted. Shape: {trigger_channel_data.shape}")

        # Set to keep track of processed JSON files to avoid reprocessing
        processed_jsons = set()
        # List to store data for all runs
        all_runs_data = []
        # List to store metadata for each run
        runs_info = []

        # Process each run based on BIDS convention
        for run_idx in range(1, 5):  # Assuming 4 runs
            run_id = f"run-{run_idx:02d}"
            json_file_name = f"{subject_id}_{session_id}_task-rest_{run_id}_bold.json"
            json_file_path = os.path.join(bids_root_dir, subject_id, session_id, 'func', json_file_name)

            # Extract run metadata from JSON file
            run_metadata = extract_metadata_from_json(json_file_path, processed_jsons)
            logging.info(f"Metadata for run {run_id} extracted successfully.")
            # After extracting run metadata from JSON file
            logging.info(f"Run metadata for {run_id}: {run_metadata}")
            if run_metadata is None:
                logging.warning(f"Metadata for run {run_id} could not be found. Skipping.")
                continue

            # Find the runs in the data using the extracted trigger data
            current_runs_info = find_runs(data, run_metadata, trigger_channel_data, sampling_rate=5000)
            # After finding runs in the data
            logging.info(f"Found {len(current_runs_info)} runs for {run_id}.")
            for run_number, run_info in enumerate(current_runs_info, start=1):
                logging.info(f"Run {run_number} for {run_id}: Start index: {run_info['start_index']}, End index: {run_info['end_index']}")
                runs_info.append(run_info)
            logging.info(f"Processing Current Run {run_id} of {len(current_runs_info)} Total Runs Identified.")

            for run_info in current_runs_info:
                segmented_data = run_info['data']
                all_runs_data.append(segmented_data)  # Append segmented data for the run
                runs_info.append(run_info)

                # Assuming the last element in runs_info pertains to the current run
                current_run_info = runs_info[-1]
                metadata_dict = create_metadata_dict(current_run_info, 5000, bids_labels_dictionary, bids_labels_list, units_dict)

            # Prepare to write output files
            output_dir = os.path.join(bids_root_dir, subject_id, session_id, 'func')
            write_output_files(segmented_data, run_metadata, metadata_dict, labels, output_dir, subject_id, session_id, run_id)
            #logging.info(f"Output files for run {run_id} written successfully.")

        # Plot physiological data for all runs
        original_labels = labels  # Assuming 'labels' are the original labels from the .mat file
        sampling_rate = 5000  # Define the sampling rate
        plot_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-rest_all_runs_physio.png")
        
        logging.info(f"Full dataset shape: {data.shape}")
        logging.info(f"Number of runs to plot: {len(runs_info)}")
        logging.info(f"Original labels being passed to plot_runs: {original_labels}")
        logging.info(f"BIDS labels being passed to plot_runs: {bids_labels_list}")

        for idx, run_info in enumerate(runs_info):
            logging.info(f"Run {idx+1} start index: {run_info['start_index']}, data shape: {run_info['data'].shape}")
            
        # Before plotting, ensure that runs_info contains only unique runs
        unique_runs_info = []
        seen_indices = set()
        for run_info in runs_info:
            start_index = run_info['start_index']
            if start_index not in seen_indices:
                unique_runs_info.append(run_info)
                seen_indices.add(start_index)
            else:
                logging.warning(f"Duplicate run detected at start index {start_index} and will be ignored.")

        runs_info = unique_runs_info
        logging.info(f"Number of unique runs to plot after deduplication: {len(runs_info)}")

        # Call the plot_runs function
        plot_runs(data, runs_info, bids_labels_list, sampling_rate, original_labels, plot_file_path)
        logging.info(f"Physiological data plotted and saved to {plot_file_path}.")

        logging.info("Main processing completed without errors.")

    except Exception as e:
        logging.error("An error occurred during the main processing", exc_info=True)
        logging.error("Processing terminated due to an unexpected error.")
        raise

# Main function to run the script from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir)