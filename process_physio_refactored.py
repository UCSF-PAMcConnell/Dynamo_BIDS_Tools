import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import argparse
import re
import matplotlib.pyplot as plt
import warnings
import gzip
import logging

# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_metadata(json_path):
    """Load metadata from a JSON file and print out TR and number of volumes."""
    print(f"Loading JSON metadata from {json_path}")
    with open(json_path, 'r') as file:
        metadata = json.load(file)
    tr = metadata.get('RepetitionTime')
    num_volumes = metadata.get('NumVolumes')
    print(f"TR (RepetitionTime): {tr}, Number of Volumes: {num_volumes}")
    return metadata

def extract_subject_session_ids(physio_root_dir):
    pattern = r'sub-(?P<subject_id>\w+)/ses-(?P<session_id>\w+)/physio'
    match = re.search(pattern, physio_root_dir)
    if not match:
        logging.error("Unable to extract subject_id and session_id from the physio_root_dir.")
        raise ValueError("Unable to extract subject_id and session_id from the physio_root_dir.")
    return match.group('subject_id'), match.group('session_id')

def load_mat_file(mat_file_path):
    try:
        mat_data = loadmat(mat_file_path)
        if 'data' not in mat_data or 'labels' not in mat_data:
            raise KeyError("The .mat file does not contain 'data' or 'labels' key.")
        return mat_data['data'], mat_data['labels']
    except FileNotFoundError:
        logging.error(f"The .mat file at {mat_file_path} was not found.")
        raise
    except KeyError as e:
        logging.error(str(e))
        raise

def extract_trigger_points(mri_trigger_data, threshold=5):
    """Extract trigger points from MRI trigger channel data."""
    triggers = (mri_trigger_data > threshold).astype(int)
    diff_triggers = np.diff(triggers, prepend=0)
    trigger_starts = np.where(diff_triggers == 1)[0]
    return trigger_starts

def find_trigger_points(data, labels, trigger_label):

    # Assume trigger_label is a string that identifies the trigger channel

    trigger_channel_indices = np.where(labels == trigger_label)[0]
    if trigger_channel_indices.size == 0:
        logging.error(f"Trigger label '{trigger_label}' not found in the labels.")
        raise ValueError(f"Trigger label '{trigger_label}' not found in the labels.")

    trigger_channel_index = trigger_channel_indices[0]
    mri_trigger_data = data[:, trigger_channel_index]

    return extract_trigger_points(mri_trigger_data)

def segment_data_into_runs(data, trigger_starts, tr, sampling_rate, num_volumes_per_run, start_from_index=0):
    """Segment the data into runs based on consecutive sequences of triggers."""
    samples_per_volume = int(sampling_rate * tr)
    runs = []
    
    # Start from the provided index and look for a set of triggers that match the expected number of volumes
    i = start_from_index
    while i < len(trigger_starts) - num_volumes_per_run + 1:
        expected_interval = samples_per_volume * (num_volumes_per_run - 1)
        actual_interval = trigger_starts[i + num_volumes_per_run - 1] - trigger_starts[i]
        
        # Debugging: Log the checking process
        #print(f"Checking triggers from index {i}: expected interval {expected_interval}, actual interval {actual_interval}")
        
        if actual_interval <= expected_interval:
            start_idx = trigger_starts[i]
            end_idx = start_idx + num_volumes_per_run * samples_per_volume
            segment = data[start_idx:end_idx, :]
            runs.append({'data': segment, 'start_index': start_idx})
            
            # Debugging: Log when a matching segment is found
            print(f"Run segmented from index {start_idx} to {end_idx}")
            return runs, i + num_volumes_per_run
        
        # If the segment does not match, increment and continue searching
        i += 1
    
    # If no matching segments are found, log that information
    print(f"No valid segments found after index {start_from_index}.")
    return [], start_from_index

def segment_runs(data, trigger_starts, json_metadata, sampling_rate):
    num_volumes_per_run = json_metadata.get('NumVolumes')
    tr = json_metadata.get('RepetitionTime')
    if num_volumes_per_run is None or tr is None:
        logging.error(f"JSON metadata does not contain 'NumVolumes' or 'RepetitionTime'.")
        raise ValueError(f"JSON metadata does not contain 'NumVolumes' or 'RepetitionTime'.")
    return segment_data_into_runs(data, trigger_starts, tr, sampling_rate, num_volumes_per_run)

def generate_and_save_metadata(physio_json_metadata, bids_root_dir, subject_id, session_id, run_idx):
    # This function will generate the appropriate JSON metadata and save it
    physio_json_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(run_idx).zfill(2)}_physio.json"
    physio_json_file_path = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func', physio_json_file_name)
    with open(physio_json_file_path, 'w') as f_json:
        json.dump(physio_json_metadata, f_json, indent=4)
    logging.info(f"Saved physio JSON metadata for run {run_idx} to {physio_json_file_path}")

def create_physio_json_metadata(new_labels, sampling_rate, start_index):
    physio_json_metadata = {
        "SamplingFrequency": {
            "Value": sampling_rate,
            "Units": "Hz",
        },
        "StartTime": {
            "Value": start_index / sampling_rate,
            "Description": "start time of current run relative to recording onset",
            "Units": "seconds",
        },
        "Columns": new_labels
    }

    # Add metadata for specific labels if they exist
    if 'cardiac' in new_labels:
        physio_json_metadata['cardiac'] = {
            "Description": "continuous ecg measurement",
            "Placement": "Lead 1",
            "Units": "mV",
            "Gain": 500,
            "35HzLPN": "off / 150HzLP",
            "HPF": "0.05 Hz",
        }

    if 'respiratory' in new_labels:
        physio_json_metadata['respiratory'] = {
            "Description": "continuous measurements by respiration belt",
            "Units": "Volts",
            "Gain": 10,
            "LPF": "10 Hz",
            "HPF1": "DC",
            "HPF2": "0.05 Hz",
        }

    if 'eda' in new_labels:
        physio_json_metadata['eda'] = {
            "Description": "continuous eda measurement",
            "Placement": "right plantar instep",
            "Units": "microsiemens",
            "Gain": 5,
            "LPF": "1.0 Hz",
            "HPF1": "DC",
            "HPF2": "DC",
        }

    if 'ppg' in new_labels:
        physio_json_metadata['ppg'] = {
            "Description": "continuous ppg measurement",
            "Placement": "left index toe",
            "Units": "Volts",
            "Gain": 10,
            "LPF": "3.0 Hz",
            "HPF1": "0.5 Hz",
            "HPF2": "0.05 Hz",
        }

    # Add more entries as needed for other labels

    return physio_json_metadata

def save_segments_to_tsv_and_json(segments, bids_root_dir, subject_id, session_id, original_labels):
    # Define the new labels according to BIDS format
    bids_labels = {
        'ECG Test - ECG100C': 'cardiac',
        'RSP Test - RSP100C': 'respiratory',
        'EDA - EDA100C-MRI': 'eda',
        'MRI Trigger - Custom, HLT100C - A 4': 'trigger'
    }
    
    # Check if 'PPG - PPG100C' is in the labels and add it to the dictionary
    if 'PPG - PPG100C' in original_labels:
        bids_labels['PPG - PPG100C'] = 'ppg'

    # Identify the columns to keep based on whether the original label is in our mapping
    columns_to_keep = [idx for idx, label in enumerate(original_labels) if label in bids_labels]
    
    # Create a new list of labels for the columns we are keeping
    new_labels = [bids_labels[original_labels[idx]] for idx in columns_to_keep]
    
    for i, segment in enumerate(segments):
        segment_data = segment['data']
        # Select only the columns we want to keep
        filtered_segment_data = segment_data[:, columns_to_keep]
        
        # Create a DataFrame with the new labels
        df_segment = pd.DataFrame(filtered_segment_data, columns=new_labels)
        
        # Prepare the output directory and filename
        output_dir = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func')
        os.makedirs(output_dir, exist_ok=True)
        filename = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(i+1).zfill(2)}_physio.tsv.gz"
        filepath = os.path.join(output_dir, filename)
        
        # Save the DataFrame as a .tsv.gz file
        with gzip.open(filepath, 'wt') as f_out:
            df_segment.to_csv(f_out, sep='\t', index=False)
        print(f"Saved {filename} to {output_dir}")

        return new_labels

def save_run_segments(segments, bids_root_dir, subject_id, session_id, filtered_labels, run_idx, sampling_rate):
    # Save the segments to TSV files using the provided function
    new_labels = save_segments_to_tsv_and_json(segments, bids_root_dir, subject_id, session_id, filtered_labels, run_idx, sampling_rate)

    # Generate the JSON metadata for the physiological data
    physio_json_metadata = create_physio_json_metadata(new_labels, sampling_rate, segments[0]['start_index'])

    # Save the JSON metadata
    physio_json_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(run_idx).zfill(2)}_physio.json"
    physio_json_file_path = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func', physio_json_file_name)
    try:
        with open(physio_json_file_path, 'w') as f_json:
            json.dump(physio_json_metadata, f_json, indent=4)
        logging.info(f"Saved physio JSON metadata for run {run_idx} to {physio_json_file_path}")
    except IOError as e:
        logging.error(f"Failed to save physio JSON metadata for run {run_idx}: {e}")
        raise

    return new_labels

def plot_full_data_with_segments(data, runs, sampling_rate, original_labels, output_fig_path):
    """Plot the full data with segments highlighted."""
    filtered_labels = [label for label in original_labels if 'Digital input' not in label]
    time = np.arange(data.shape[0]) / sampling_rate
    num_subplots = len(filtered_labels)

    fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 10), sharex=True)
    if num_subplots == 1:
        axes = [axes]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for i, label in enumerate(filtered_labels):
            axes[i].plot(time, data[:, original_labels.index(label)], label=f"Full Data {label}", color='lightgrey')
            for run_idx, run in enumerate(runs):
                start_time_of_run = run['start_index'] / sampling_rate
                run_length = run['data'].shape[0]
                run_time = start_time_of_run + np.arange(run_length) / sampling_rate
                axes[i].plot(run_time, run['data'][:, original_labels.index(label)], label=f"Run {run_idx+1} {label}")
            axes[i].legend(loc='upper right')

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    plt.savefig(output_fig_path, dpi=300)
    print(f"Saved plot to {output_fig_path}")

def plot_and_save_data(data, all_runs, sampling_rate, labels, output_fig_path):
    # This function will call plot_full_data_with_segments
    plot_full_data_with_segments(data, all_runs, sampling_rate, labels, output_fig_path)
    logging.info(f"Saved plot to {output_fig_path}")

def check_json_file_exists(bids_root_dir, subject_id, session_id, run_idx):
    json_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(run_idx).zfill(2)}_bold.json"
    json_file_path = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func', json_file_name)
    if not os.path.isfile(json_file_path):
        logging.info(f"No JSON file found for run {str(run_idx).zfill(2)}. Ending search for runs.")
        return False, None
    return True, json_file_path

def process_and_save_run(data, trigger_starts, bids_root_dir, subject_id, session_id, run_idx, sampling_rate, filtered_labels):
    json_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(run_idx).zfill(2)}_bold.json"
    json_file_path = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func', json_file_name)

    json_metadata = load_json_metadata(json_file_path)
    runs, next_start_index = segment_data_into_runs(data, trigger_starts, json_metadata['RepetitionTime'], sampling_rate, json_metadata['NumVolumes'])

    if runs:
        # Save the segments to TSV and JSON
        try:
            new_labels = save_run_segments(runs, bids_root_dir, subject_id, session_id, filtered_labels, run_idx, sampling_rate)
            logging.info(f"Processed and saved run {run_idx}")
        except Exception as e:
            logging.error(f"Failed to save segments for run {run_idx}: {e}")
            raise
        return next_start_index
    else:
        logging.info(f"No valid segments found for run {str(run_idx).zfill(2)}.")
        return None

def main(physio_root_dir, bids_root_dir, filtered_labels):
    subject_id, session_id = extract_subject_session_ids(physio_root_dir)
    mat_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_physio.mat"
    mat_file_path = os.path.join(physio_root_dir, mat_file_name)
    data, labels = load_mat_file(mat_file_path)

    trigger_label = 'MRI Trigger - Custom, HLT100C - A 4'
    trigger_starts = find_trigger_points(data, labels, trigger_label)

    all_runs = []
    run_idx = 1
    start_from_trigger_index = 0
    sampling_rate = 5000  # Hz, set as a constant in this script
    
    # Main processing loop
    while True:
        # Check for the existence of the JSON file for the current run
        json_exists, json_file_path = check_json_file_exists(bids_root_dir, subject_id, session_id, run_idx)
        if not json_exists:
            break  # No more runs to process

        # Process and save the run segments and metadata
        next_start_index = process_and_save_run(data, trigger_starts, bids_root_dir, subject_id, session_id, run_idx, sampling_rate, filtered_labels)
        if next_start_index is None:
            break  # No valid segments found, stop processing

        # Prepare for the next run
        start_from_trigger_index = next_start_index
        run_idx += 1

    # Plot and save data with segments highlighted
    output_fig_path = os.path.join(physio_root_dir, f"{mat_file_name.replace('.mat', '')}_all_runs.png")
    
    plot_and_save_data(data, all_runs, sampling_rate, labels, output_fig_path)
    logging.info("Processing complete.")

# Main code execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir, filtered_labels=[])
