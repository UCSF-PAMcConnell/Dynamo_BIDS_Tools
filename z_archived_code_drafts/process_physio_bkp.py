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

# Define a function to load metadata from a JSON file
def load_json_metadata(json_path):
    with open(json_path, 'r') as file:
        metadata = json.load(file)
    return metadata

# Define a function to extract trigger points from MRI trigger channel data
def extract_trigger_points(mri_trigger_data, threshold=5):
    triggers = (mri_trigger_data > threshold).astype(int)
    diff_triggers = np.diff(triggers, prepend=0)
    trigger_starts = np.where(diff_triggers == 1)[0]
    return trigger_starts

# Define a function to segment the data into runs based on consecutive sequences of triggers
def segment_data_into_runs(data, trigger_starts, tr, sampling_rate, num_volumes_per_run=175):
    samples_per_volume = int(sampling_rate * tr)
    runs = []
    current_run = []
    for i in range(len(trigger_starts) - 1):
        if len(current_run) < num_volumes_per_run:
            current_run.append(trigger_starts[i])
        if len(current_run) == num_volumes_per_run or trigger_starts[i+1] - trigger_starts[i] > samples_per_volume:
            if len(current_run) == num_volumes_per_run:
                start_idx = current_run[0]
                # Ensure the end index includes the last sample of the last volume
                end_idx = start_idx + num_volumes_per_run * samples_per_volume
                segment = data[start_idx:end_idx, :]
                runs.append({'data': segment, 'start_index': start_idx})
            current_run = []
    # Check for any remaining triggers that might form a run
    if len(current_run) == num_volumes_per_run:
        start_idx = current_run[0]
        end_idx = start_idx + num_volumes_per_run * samples_per_volume
        segment = data[start_idx:end_idx, :]
        runs.append({'data': segment, 'start_index': start_idx})
    return runs


# Define a function to save the segments into TSV files, excluding unwanted labels
import gzip

def save_segments_to_tsv(segments, bids_root_dir, subject_id, session_id, original_labels):
    # Define the new labels according to BIDS format
    bids_labels = {
        'ECG Test - ECG100C': 'cardiac',
        'RSP Test - RSP100C': 'respiratory',
        'EDA - EDA100C-MRI': 'eda',
        'MRI Trigger - Custom, HLT100C - A 4': 'trigger'
    }
    
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

# Define a function to plot the full data with segments highlighted
def plot_full_data_with_segments(data, runs, sampling_rate, original_labels, output_fig_path):
    filtered_labels = [label for label in original_labels if 'Digital input' not in label]
    time = np.arange(data.shape[0]) / sampling_rate
    num_subplots = len(filtered_labels)
    
    fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 10), sharex=True)
    if num_subplots == 1:
        axes = [axes]  # Make sure axes is always a list, even for one subplot
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for i, label in enumerate(filtered_labels):
            axes[i].plot(time, data[:, original_labels.index(label)], label=f"Full Data {label}", color='lightgrey')
            for run_idx, run in enumerate(runs):
                # Calculate the actual start time for the current run segment
                start_time_of_run = run['start_index'] / sampling_rate
                run_length = run['data'].shape[0]
                run_time = start_time_of_run + np.arange(run_length) / sampling_rate
                
                # Plot the run segment with the correct time offset
                axes[i].plot(run_time, run['data'][:, original_labels.index(label)], label=f"Run {run_idx+1} {label}")
            axes[i].legend(loc='upper right')
    
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    # Save the figure
    plt.savefig(output_fig_path, dpi=300)
    #plt.close()  # Close the figure to free memory

# Define the main function to process and convert physiological data to BIDS format
def main(physio_root_dir, bids_root_dir):
    pattern = r'sub-(?P<subject_id>\w+)/ses-(?P<session_id>\w+)/physio'
    match = re.search(pattern, physio_root_dir)
    if not match:
        raise ValueError("Unable to extract subject_id and session_id from the physio_root_dir.")
    subject_id = match.group('subject_id')
    session_id = match.group('session_id')
    mat_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_physio.mat"
    mat_file_path = os.path.join(physio_root_dir, mat_file_name)
    mat_data = loadmat(mat_file_path)
    if 'data' not in mat_data:
        raise KeyError("The .mat file does not contain 'data' key.")
    data = mat_data['data']
    if 'labels' not in mat_data:
        raise KeyError("The .mat file does not contain 'labels' key.")
    labels_struct = mat_data['labels']
    labels = [str(label.flat[0]) for label in labels_struct]
    sampling_rate = 5000 # Hz, set as a constant in this script
    trigger_label = 'MRI Trigger - Custom, HLT100C - A 4'
    if trigger_label not in labels:
        raise ValueError(f"Trigger label '{trigger_label}' not found in the labels. Available labels: {labels}")
    trigger_channel_index = labels.index(trigger_label)
    mri_trigger_data = data[:, trigger_channel_index]
    trigger_starts = extract_trigger_points(mri_trigger_data)
    runs = segment_data_into_runs(data, trigger_starts, 2, sampling_rate)
    
    # Generate the path for the output figure based on the .mat file name
    output_fig_path = os.path.join(physio_root_dir, mat_file_name.replace('.mat', '.png'))
    
    # Call the plotting function with the path to save the figure
    plot_full_data_with_segments(data, runs, sampling_rate, labels, output_fig_path)
    #plot_full_data_with_segments(data, runs, sampling_rate, labels)
    save_segments_to_tsv(runs, bids_root_dir, subject_id, session_id, labels)

# Command-line interface setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir)
