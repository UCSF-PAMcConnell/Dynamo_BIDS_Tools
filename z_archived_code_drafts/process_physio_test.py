import os
import json
import argparse
import scipy.io
import numpy as np

def extract_labels(mat_contents):
    """
    Extracts labels from the .mat file contents.
    
    Parameters:
    - mat_contents: The dictionary containing .mat file contents.
    
    Returns:
    A list of label strings.
    """
    raw_labels = mat_contents['labels']
    # Check if the labels are stored as an array of arrays with a single string
    if isinstance(raw_labels[0], np.ndarray) and raw_labels[0].size == 1:
        labels = [label[0] for label in raw_labels]
    # Check if the labels are stored directly as an array of strings
    elif isinstance(raw_labels[0], str):
        labels = list(raw_labels)
    # Add more conditions if there are different structures of labels in the .mat files
    else:
        raise ValueError(f"Unexpected format of labels in .mat file: {raw_labels}")
    return labels


def read_json_file(json_file_path):
    """
    Read the JSON file and return its contents.
    
    Parameters:
    - json_file_path: The path to the JSON file.
    
    Returns:
    - The contents of the JSON file as a dictionary.
    """
    with open(json_file_path, 'r') as file:
        return json.load(file)

def segment_and_save_data(data, triggers, num_volumes, repetition_times, physio_root_dir, labels, sampling_rate):
    """
    Segment the data based on MR triggers and save to BIDS-compliant TSV files.
    
    Parameters:
    - data: The continuous physiological data array.
    - triggers: The indices of the MR triggers.
    - num_volumes: The number of volumes in the fMRI run.
    - repetition_time: The repetition time for the fMRI run.
    - output_path: The path where the segmented files will be saved.
    - labels: The list of labels for the physiological data channels.
    """
    # Calculate the number of samples per volume
    samples_per_volume = int(round(repetition_time / (1 / sampling_rate)))

    for i, trigger in enumerate(triggers):
        # Calculate the start and end points of the segment
        start_point = trigger
        end_point = trigger + num_volumes[i] * samples_per_volume

        # Segment the data
        segmented_data = data[start_point:end_point, :]

        # Convert the segmented data to a list of lists (suitable for saving in TSV format)
        segmented_data_list = segmented_data.tolist()

        # Define the output filename based on BIDS convention
        output_filename = f"sub-LRN001_ses-1_task-rest_run-{str(i+1).zfill(2)}_physio.tsv"
        output_filepath = os.path.join(output_path, output_filename)

        # Write the segmented data to a TSV file
        with open(output_filepath, 'w') as tsv_file:
            tsv_file.write('\t'.join(labels) + '\n')
            for data_row in segmented_data_list:
                tsv_file.write('\t'.join(map(str, data_row)) + '\n')
        
        print(f"Saved segmented data to {output_filepath}")

def main(physio_root_dir, bids_root_dir):
    """
    Main function to process the physiological data and convert to BIDS format.
    
    Parameters:
    - physio_root_dir: The directory containing the raw physiological .mat file.
    - bids_root_dir: The root directory of the BIDS dataset.
    """
    # Define the .mat file path (assumes the .mat file is named following a specific convention)
    mat_file_path = os.path.join(physio_root_dir, 'sub-LRN001_ses-1_task-rest_physio.mat')
    
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_file_path, simplify_cells=True)
    
    # Extract labels and data
    labels = extract_labels(mat_contents)
    data = mat_contents['data']
    sampling_rate = 1 / mat_contents['isi']  # Directly access the 'isi' as a float

    # Find the MRI trigger channel index
    trigger_label = 'MRI Trigger - Custom, HLT100C - A 4'
    if trigger_label not in labels:
        raise ValueError(f"Trigger label '{trigger_label}' not found in the labels. Available labels: {labels}")
    trigger_channel_index = labels.index(trigger_label)
    
    # Extract the trigger points from the data
    trigger_data = data[:, trigger_channel_index]
    triggers = np.where(trigger_data > 6)[0]  # Threshold value for trigger detection

    # Read JSON files and collect metadata for each fMRI run
    num_volumes = []
    repetition_times = []
    for run_num in range(1, 5):  # Assumes there are 4 runs
        json_file_path = os.path.join(bids_root_dir, 'sub-LRN001', 'ses-1', 'func', f'sub-LRN001_ses-1_task-rest_run-{str(run_num).zfill(2)}_bold.json')
        json_contents = read_json_file(json_file_path)
        num_volumes.append(json_contents['NumVolumes'])
        repetition_times.append(json_contents['RepetitionTime'])

    # Segment and save the data
    segment_and_save_data(data, triggers, num_volumes, repetition_times, physio_root_dir, labels, sampling_rate)

# Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process physiological data for BIDS.")
    parser.add_argument('physio_root_dir', help='The directory of the raw physio .mat file')
    parser.add_argument('bids_root_dir', help='The BIDS dataset root directory')
    
    args = parser.parse_args()
    
    # Define the constant sampling rate
    sampling_rate = 5000  # in Hz
    
    main(args.physio_root_dir, args.bids_root_dir)
