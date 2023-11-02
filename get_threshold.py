import argparse
import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt

def read_mat_file(physio_root_dir, mat_file_name):
    """
    Load the .mat file containing the physiological data.
    
    Parameters:
    - physio_root_dir: The directory containing the physiological data file.
    - mat_file_name: The name of the .mat file to read.

    Returns:
    - mat_contents: A dictionary with the contents of the .mat file.
    """
    mat_file_path = os.path.join(physio_root_dir, mat_file_name)
    mat_contents = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    return mat_contents

def plot_trigger_channel(trigger_channel, num_samples=1000):
    """
    Plot the first `num_samples` data points of the MR trigger channel
    to help determine the appropriate threshold for detecting triggers.
    
    Parameters:
    - trigger_channel: An array containing the trigger channel data.
    - num_samples: The number of samples to plot (default is 1000).
    """
    plt.figure(figsize=(12, 4))
    plt.plot(trigger_channel[:num_samples])
    plt.title('MR Trigger Channel')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.show()

def extract_subject_session_ids(physio_root_dir):
    """
    Extract subject and session IDs from the physio root directory path.
    
    Parameters:
    - physio_root_dir: The directory containing the physiological data file.
    
    Returns:
    - subject_id: The subject identifier.
    - session_id: The session identifier.
    """
    path_parts = physio_root_dir.strip(os.sep).split(os.sep)
    subject_id = next(part for part in path_parts if part.startswith('sub-'))
    session_id = next(part for part in path_parts if part.startswith('ses-'))
    return subject_id, session_id

def main(physio_root_dir, bids_root_dir):
    """
    The main function to process the physiological data and convert it into BIDS format.
    
    Parameters:
    - physio_root_dir: The directory containing the physiological data file.
    - bids_root_dir: The root directory path of the BIDS dataset.
    """
    # Extract subject_id and session_id from the physio_root_dir
    subject_id, session_id = extract_subject_session_ids(physio_root_dir)
    
    # Construct the .mat file name based on subject_id and session_id
    mat_file_name = f"{subject_id}_{session_id}_task-rest_physio.mat"

    # Load the MATLAB file
    mat_contents = read_mat_file(physio_root_dir, mat_file_name)

    # Extract the labels
    labels = mat_contents['labels']

    # Find the index of the MRI trigger channel
    trigger_label = 'MRI Trigger - Custom, HLT100C - A 4'
    trigger_channel_index = labels.tolist().index(trigger_label)

    # Extract the trigger channel data
    data = mat_contents['data']
    trigger_channel = data[:, trigger_channel_index]

    # Plot the trigger channel to determine the threshold visually
    plot_trigger_channel(trigger_channel)

    # The script will continue to process the data once the threshold is determined

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process physio data for BIDS.")
    parser.add_argument("physio_root_dir", help="Directory path to the raw physio data (.mat files)")
    parser.add_argument("bids_root_dir", help="Root directory path of the BIDS dataset")
    
    args = parser.parse_args()
    
    main(args.physio_root_dir, args.bids_root_dir)
