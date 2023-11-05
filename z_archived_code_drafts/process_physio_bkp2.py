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

def load_json_metadata(json_path):
    """Load metadata from a JSON file and print out TR and number of volumes."""
    print(f"Loading JSON metadata from {json_path}")
    with open(json_path, 'r') as file:
        metadata = json.load(file)
    tr = metadata.get('RepetitionTime')
    num_volumes = metadata.get('NumVolumes')
    print(f"TR (RepetitionTime): {tr}, Number of Volumes: {num_volumes}")
    return metadata

def extract_trigger_points(mri_trigger_data, threshold=5):
    """Extract trigger points from MRI trigger channel data."""
    triggers = (mri_trigger_data > threshold).astype(int)
    diff_triggers = np.diff(triggers, prepend=0)
    trigger_starts = np.where(diff_triggers == 1)[0]
    return trigger_starts

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

def save_segments_to_tsv_and_json(segments, bids_root_dir, subject_id, session_id, original_labels, run_index, sampling_rate):
    """Save the segments into TSV files and corresponding JSON metadata files, excluding unwanted labels."""
    bids_labels = {
         'ECG Test - ECG100C': 'cardiac',
         'RSP Test - RSP100C': 'respiratory',
         'EDA - EDA100C-MRI': 'eda',
         'MRI Trigger - Custom, HLT100C - A 4': 'trigger'
     }
    # Check if 'PPG - PPG100C' is in the labels and add it to the dictionary
    if 'PPG - PPG100C' in original_labels:
        bids_labels['PPG - PPG100C'] = 'ppg'
   
    columns_to_keep = [idx for idx, label in enumerate(original_labels) if label in bids_labels]
    new_labels = [bids_labels[original_labels[idx]] for idx in columns_to_keep]
    segment_data = segments[0]['data']
    filtered_segment_data = segment_data[:, columns_to_keep]
    df_segment = pd.DataFrame(filtered_segment_data, columns=new_labels)
    output_dir = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(run_index).zfill(2)}_physio.tsv.gz"
    filepath = os.path.join(output_dir, filename)
    with gzip.open(filepath, 'wt') as f_out:
        df_segment.to_csv(f_out, sep='\t', index=False)
    print(f"Saved segment for run {run_index} to {output_dir} with filename {filename}")

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

def main(physio_root_dir, bids_root_dir, filtered_labels):
    """Process and convert physiological data to BIDS format."""
    pattern = r'sub-(?P<subject_id>\w+)/ses-(?P<session_id>\w+)/physio'
    match = re.search(pattern, physio_root_dir)
    if not match:
        raise ValueError("Unable to extract subject_id and session_id from the physio_root_dir.")
    subject_id = match.group('subject_id')
    session_id = match.group('session_id')
    print(f"Processing subject: {subject_id}, session: {session_id}")
    mat_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_physio.mat"
    mat_file_path = os.path.join(physio_root_dir, mat_file_name)
    print(f"Loading data from {mat_file_path}")
    mat_data = loadmat(mat_file_path)
    if 'data' not in mat_data or 'labels' not in mat_data:
        raise KeyError("The .mat file does not contain 'data' or 'labels' key.")
    data = mat_data['data']
    labels_struct = mat_data['labels']
    labels = [str(label.flat[0]) for label in labels_struct]
    sampling_rate = 5000  # Hz, set as a constant in this script
    trigger_label = 'MRI Trigger - Custom, HLT100C - A 4'
    if trigger_label not in labels:
        raise ValueError(f"Trigger label '{trigger_label}' not found in the labels.")
    trigger_channel_index = labels.index(trigger_label)
    mri_trigger_data = data[:, trigger_channel_index]
    trigger_starts = extract_trigger_points(mri_trigger_data)

    # Main processing loop
    all_runs = []
    run_idx = 1
    start_from_trigger_index = 0

    while True:
        print(f"Searching for runs starting from trigger index {start_from_trigger_index}")
        json_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(run_idx).zfill(2)}_bold.json"
        json_file_path = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func', json_file_name)
        if not os.path.exists(json_file_path):
            print(f"No JSON file found for run {str(run_idx).zfill(2)}. Ending search for runs.")
            break
        print(f"Processing run: sub-{subject_id}, ses-{session_id}, run-{str(run_idx).zfill(2)}")
        json_metadata = load_json_metadata(json_file_path)
        num_volumes_per_run = json_metadata.get('NumVolumes')
        tr = json_metadata.get('RepetitionTime')
        if num_volumes_per_run is None or tr is None:
            raise ValueError(f"JSON metadata for run {run_idx} does not contain 'NumVolumes' or 'RepetitionTime'.")
        runs, next_start_index = segment_data_into_runs(data, trigger_starts, tr, sampling_rate, num_volumes_per_run, start_from_trigger_index)
        if runs:
            current_run_segment = runs[0]
            all_runs.append(current_run_segment)

            # Save the segment to TSV and then save the corresponding JSON metadata
            save_segments_to_tsv_and_json([current_run_segment], bids_root_dir, subject_id, session_id, filtered_labels, run_idx, sampling_rate)

            # Save the corresponding JSON metadata for the physio
            physio_json_metadata = {
                "SamplingFrequency": {
                    "Value": sampling_rate,
                    "Units": "Hz",
                },
                "StartTime": {
                    "Value": current_run_segment['start_index'] / sampling_rate,
                    "Description": "start time of current run relative to recording onset",
                    "Units": "seconds",
                },
                "Columns": filtered_labels,
                "Manufacturer": "Biopac",
                "cardiac": {
                    "Description": "continuous ecg measurement",
                    "Placement": "Lead 1",
                    "Units": "mV",
                    "Gain": 500,
                    "35HzLPN": "off / 150HzLP",
                    "HPF": "0.05 Hz",
                }
                if 'cardiac' in filtered_labels
                else None,
                "respiratory": {
                    "Description": "continuous measurements by respiration belt",
                    "Units": "Volts",
                    "Gain": 10,
                    "LPF": "10 Hz",
                    "HPF1": "DC",
                    "HPF2": "0.05 Hz",
                }
                if 'respiratory' in filtered_labels
                else None,
                "eda": {
                    "Description": "continuous eda measurement",
                    "Placement": "right plantar instep",
                    "Units": "microsiemens",
                    "Gain": 5,
                    "LPF": "1.0 Hz",
                    "HPF1": "DC",
                    "HPF2": "DC",
                }
                if 'eda' in filtered_labels
                else None,
                "ppg": {
                    "Description": "continuous ppg measurement",
                    "Placement": "left index toe",
                    "Units": "Volts",
                    "Gain": 10,
                    "LPF": "3.0 Hz",
                    "HPF1": "0.5 Hz",
                    "HPF2": "0.05 Hz",
                }
                if 'ppg' in filtered_labels
                else None,
            }
            physio_json_file_name = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(run_idx).zfill(2)}_physio.json"
            physio_json_file_path = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func', physio_json_file_name)
            with open(physio_json_file_path, 'w') as f_json:
                json.dump(physio_json_metadata, f_json, indent=4)
            print(f"Saved physio JSON metadata for run {run_idx} to {physio_json_file_path}")
            start_from_trigger_index = next_start_index
        else:
            print(f"No valid segments found for run {str(run_idx).zfill(2)}. Ending search for runs.")
            break
        run_idx += 1  # Increment run index for next iteration
    output_fig_path = os.path.join(physio_root_dir, f"{mat_file_name.replace('.mat', '')}_all_runs.png")
    plot_full_data_with_segments(data, all_runs, sampling_rate, labels, output_fig_path)
    print("Processing complete.")

# Main code execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert physiological data to BIDS format.")
    parser.add_argument("physio_root_dir", help="Directory containing the physiological .mat file.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    main(args.physio_root_dir, args.bids_root_dir, filtered_labels=['cardiac', 'respiratory', 'eda', 'ppg'])
