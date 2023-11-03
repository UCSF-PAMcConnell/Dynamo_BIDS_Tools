import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import argparse
import re
import matplotlib.pyplot as plt
import warnings
import logging

# Configure logging
logging.basicConfig(
    filename='process_physio.log',  # Log to this file
    filemode='a',  # Append to the log file, 'w' would overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
    level=logging.DEBUG  # Set logging level to DEBUG
)

# Define a function to load metadata from a JSON file
def load_json_metadata(json_path):
    # Log the attempt to load JSON metadata.
    logging.info(f'Loading JSON metadata from {json_path}')
    
    try:
        # Open the JSON file and load its contents.
        with open(json_path, 'r') as file:
            metadata = json.load(file)
            # Log successful loading of metadata.
            logging.info(f'Successfully loaded metadata from {json_path}')
    except FileNotFoundError:
        # Log an error if the file is not found and raise the exception to the caller.
        logging.error(f'File not found: {json_path}', exc_info=True)
        raise
    except json.JSONDecodeError:
        # Log an error if the JSON file is not properly formatted and raise the exception to the caller.
        logging.error(f'Invalid JSON format in file: {json_path}', exc_info=True)
        raise
    except Exception as e:
        # Log any other exceptions that occur and raise the exception to the caller.
        logging.error(f'An unexpected error occurred while loading {json_path}: {e}', exc_info=True)
        raise
    
    # Return the metadata from the JSON file.
    return metadata

# Define a function to extract trigger points from MRI trigger channel data
def extract_trigger_points(mri_trigger_data, threshold=5):
    # Log the entry into the function and its parameters
    logging.info(f'Extracting trigger points with threshold: {threshold}')
    
    # Convert the MRI trigger data to binary based on the threshold
    # 1 if the data point is above the threshold, 0 otherwise
    triggers = (mri_trigger_data > threshold).astype(int)
    
    # Calculate the difference between successive trigger points to find rising edges
    diff_triggers = np.diff(triggers, prepend=0)
    
    # Find the indices of the start of trigger points (where the difference is 1)
    trigger_starts = np.where(diff_triggers == 1)[0]
    
    # Log the result before returning
    logging.info(f'Found {len(trigger_starts)} trigger points')
    
    # Check for empty trigger_starts, which would indicate no triggers were found
    if len(trigger_starts) == 0:
        logging.warning('No trigger points found. Check the threshold and input data.')

    return trigger_starts

# Define a function to segment the data into runs based on consecutive sequences of triggers
def segment_data_into_runs(data, trigger_starts, tr, sampling_rate, num_volumes):
    # Log the beginning of the segmentation process.
    logging.info('Segmenting data into runs...')

    # Calculate the number of samples that one volume spans.
    samples_per_volume = int(sampling_rate * tr)
    logging.debug(f'Samples per volume: {samples_per_volume}')

    # Initialize a list to hold the segmented runs.
    runs = []

    # Start at the first trigger point.
    i = 0

    # Continue processing as long as there are trigger starts left.
    while i < len(trigger_starts):
        # Get the current trigger start index.
        start = trigger_starts[i]
        logging.debug(f'Processing trigger start at index {i}: {start}')

        # If there aren't enough triggers left to form a complete run, exit the loop.
        if i + num_volumes > len(trigger_starts):
            logging.warning('Not enough triggers left to form a full run.')
            break

        # Calculate the trigger index for the end of the run.
        end_trigger = trigger_starts[i + num_volumes - 1]

        # Calculate the expected end index of the data based on the number of volumes.
        expected_end_idx = start + (num_volumes * samples_per_volume)
        logging.debug(f'Expected end index for run: {expected_end_idx}')

        # Check if the actual end trigger is within one volume's worth of samples of the expected end.
        if abs(end_trigger - expected_end_idx) < samples_per_volume:
            # Extract the run data from the overall dataset using the start and expected end indexes.
            run_data = data[start:expected_end_idx, :]

            # Append the run data to the list of runs.
            runs.append({'data': run_data, 'start_index': start})
            logging.info(f'Run found and appended. Start: {start}, End: {expected_end_idx}')

            # Move the index to the start of the next potential run.
            i += num_volumes
        else:
            # If the triggers do not align with a run, move to the next trigger.
            logging.debug(f'Triggers do not align at index {i}. Skipping to next trigger.')
            i += 1

    # Log the completion of the segmentation process and the number of runs found.
    logging.info('Segmentation complete. Total runs found: {}'.format(len(runs)))

    # Return the list of segmented runs.
    return runs

# Define a function to save the segments into TSV files, excluding unwanted labels
def save_segments_to_tsv(segments, bids_root_dir, subject_id, session_id, original_labels):
    # Map the original labels to the BIDS-compliant labels
    bids_labels = {
        'ECG Test - ECG100C': 'cardiac',
        'RSP Test - RSP100C': 'respiratory',
        'EDA - EDA100C-MRI': 'eda',
        'MRI Trigger - Custom, HLT100C - A 4': 'trigger'
    }
    
    # Identify the indices of columns with labels we want to keep
    columns_to_keep = [idx for idx, label in enumerate(original_labels) if label in bids_labels]
    # Create a new list of labels for the columns we are keeping
    new_labels = [bids_labels[original_labels[idx]] for idx in columns_to_keep]
    
    # Create the output directory if it doesn't exist
    func_dir = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func')
    os.makedirs(func_dir, exist_ok=True)
    
    # Log the start of the saving process
    logging.info(f"Saving segments to TSV files in directory: {func_dir}")
    
    for i, segment in enumerate(segments):
        # Create a DataFrame for the current segment with the new labels
        df_segment = pd.DataFrame(segment['data'][:, columns_to_keep], columns=new_labels)
        
        # Prepare the filename and filepath using BIDS naming conventions
        filename = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(i+1).zfill(2)}_physio.tsv.gz"
        filepath = os.path.join(func_dir, filename)
        
        # Save the DataFrame as a gzipped TSV file
        df_segment.to_csv(filepath, sep='\t', index=False, compression='gzip')
        
        # Log the successful save action
        logging.info(f"Saved {filename} to {func_dir}")
        
        # Optionally, you can still print to stdout if required
        print(f"Saved {filename} to {func_dir}")

    # Log the completion of the save process
    logging.info("All segments have been saved to TSV files.")

# Define a function to plot the full data with segments highlighted
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging

def plot_full_data_with_segments(data, runs, sampling_rate, original_labels, output_fig_path):
    # Log the start of the plotting process
    logging.info('Starting the plotting of full data with segments.')
    
    # Filter out labels that contain 'Digital input'
    filtered_labels = [label for label in original_labels if 'Digital input' not in label]
    
    # Calculate time points based on the sampling rate and data length
    time = np.arange(data.shape[0]) / sampling_rate
    
    # Determine the number of subplots based on the number of filtered labels
    num_subplots = len(filtered_labels)
    
    # Create subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 10), sharex=True)
    # Ensure axes is a list even if there is only one subplot
    axes = axes if num_subplots > 1 else [axes]
    
    # Suppress any warnings that arise during plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        
        # Plot each label in its own subplot
        for i, label in enumerate(filtered_labels):
            label_index = original_labels.index(label)
            
            # Plot the full data for the current label
            axes[i].plot(time, data[:, label_index], label=f"Full Data {label}", color='lightgrey')
            
            # Overlay run segments for the current label
            for run_idx, run in enumerate(runs):
                # Calculate the time points for the current run segment
                run_time = time[run['start_index']:run['start_index'] + run['data'].shape[0]]
                # Plot the run segment
                axes[i].plot(run_time, run['data'][:, label_index], label=f"Run {run_idx+1} {label}")
            
            # Add the legend to the subplot
            axes[i].legend(loc='upper right')
    
    # Label the x-axis
    plt.xlabel('Time (s)')
    
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure to the specified path
    plt.savefig(output_fig_path, dpi=300)
    
    # Log the completion of the plot and its saving
    logging.info(f'Plot saved to {output_fig_path}')
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Log that the plot is closed
    logging.info('Plotting complete and figure closed.')


# Define the main function to process and convert physiological data to BIDS format
def main(physio_root_dir, bids_root_dir):
    logging.info("Starting the conversion process.")

    # Extract subject and session IDs from the directory path
    pattern = r'sub-(?P<subject_id>\w+)/ses-(?P<session_id>\w+)/physio'
    match = re.search(pattern, physio_root_dir)
    if not match:
        error_msg = "Unable to extract subject_id and session_id from the physio_root_dir."
        logging.error(error_msg)
        raise ValueError(error_msg)
    subject_id, session_id = match.groups()
    logging.info(f"Extracted subject ID: {subject_id}, session ID: {session_id}.")

    # Locate the function directory
    func_dir = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func')
    logging.info(f"Searching for JSON files in {func_dir}.")
    json_files = sorted([f for f in os.listdir(func_dir) if f.endswith('_bold.json')])

    # Process each JSON file
    for json_file in json_files:
        json_file_path = os.path.join(func_dir, json_file)
        logging.info(f"Processing metadata file: {json_file_path}.")
        try:
            with open(json_file_path, 'r') as file:
                run_metadata = json.load(file)
        except Exception as e:
            logging.error(f"Error reading {json_file_path}: {e}")
            raise

        # Extract TR and number of volumes from metadata
        tr = run_metadata.get('RepetitionTime')
        num_volumes = run_metadata.get('NumVolumes')
        if tr is None or num_volumes is None:
            error_msg = f"Metadata in {json_file} does not contain 'RepetitionTime' or 'NumVolumes'."
            logging.error(error_msg)
            raise KeyError(error_msg)

        # Load the MATLAB file
        mat_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_physio.mat"
        mat_file_path = os.path.join(physio_root_dir, mat_file_name)
        logging.info(f"Loading MATLAB file: {mat_file_path}.")
        try:
            mat_data = loadmat(mat_file_path)
        except FileNotFoundError:
            logging.error(f"The .mat file was not found at {mat_file_path}.")
            raise FileNotFoundError(f"The .mat file was not found at {mat_file_path}")

        # Validate contents of the MATLAB file
        if 'data' not in mat_data or 'labels' not in mat_data:
            error_msg = f"The .mat file {mat_file_name} does not contain 'data' and/or 'labels' keys."
            logging.error(error_msg)
            raise KeyError(error_msg)

        # Process the data and labels
        data, labels_struct = mat_data['data'], mat_data['labels']
        labels = [str(label.flat[0]) for label in labels_struct]
        logging.info("Data and labels loaded successfully.")

        # Set a fixed sampling rate (or extract from JSON/meta if variable)
        sampling_rate = 5000
        logging.info("Using a fixed sampling rate of 5000 Hz.")

        # Extract triggers and segment data into runs
        trigger_label = 'MRI Trigger - Custom, HLT100C - A 4'
        if trigger_label not in labels:
            error_msg = f"Trigger label '{trigger_label}' not found in the labels."
            logging.error(error_msg)
            raise ValueError(error_msg)

        trigger_channel_index = labels.index(trigger_label)
        trigger_starts = extract_trigger_points(data[:, trigger_channel_index])
        runs = segment_data_into_runs(data, trigger_starts, tr, sampling_rate, num_volumes)
        logging.info(f"Found {len(runs)} runs.")

        # Plot and save the segmented data
        output_fig_path = mat_file_path.replace('.mat', '.png')
        plot_full_data_with_segments(data, runs, sampling_rate, labels, output_fig_path)
        logging.info(f"Saved plot to {output_fig_path}.")

        # Save the runs as TSV files
        save_segments_to_tsv(runs, bids_root_dir, subject_id, session_id, labels)
        logging.info("All segments saved to TSV files.")

    logging.info("Conversion process completed.")
        
# Command-line interface setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat files.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset where the .json files are located.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir)
