import scipy.io
import pandas as pd
import argparse
import os
import glob
import logging

# Configuring logging to display informational messages and errors
logging.basicConfig(level=logging.INFO)

def load_matlab_data(matlab_file_path):
    """
    Load MATLAB data from the provided file path and extract relevant fields.
    """
    try:
        mat_data = scipy.io.loadmat(matlab_file_path)
        trial_events = mat_data['trialEvents']
        block_data = mat_data['blockData'][0, 0]
        return trial_events, block_data
    except KeyError as e:
        logging.error(f"Missing necessary key in MATLAB file {matlab_file_path}: {e}")
        return None, None

def format_data_for_bids(trial_events, block_data):
    """
    Format the MATLAB data into a pandas DataFrame suitable for BIDS .tsv.
    """
    try:
        onset_times_ms = trial_events['trialStart'][0, 0].flatten()
        durations_ms = block_data['trialDuration'][0, 0].flatten()
        trial_types_ind = block_data['sequenceInd'][0, 0].flatten()
        trial_types = ['sequence' if ind == 1 else 'random' for ind in trial_types_ind]
        
        data = pd.DataFrame({
            'onset': onset_times_ms,
            'duration': durations_ms,
            'trial_type': trial_types
        })
        return data
    except KeyError as e:
        logging.error(f"Missing necessary key in data structure: {e}")
        return None

def save_as_tsv(data, output_path):
    """
    Save the formatted DataFrame as a .tsv file at the specified output path.
    """
    if data is not None:
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        data.to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process behavioral files and convert to .tsv in BIDS format.')
    parser.add_argument('matlab_dir', type=str, help='Directory containing the MATLAB files.')
    parser.add_argument('bids_root_dir', type=str, help='Root directory of the BIDS dataset where .tsv files will be saved.')
    args = parser.parse_args()
    
    # Define the order in which runs should appear
    run_order = [
        ("localizer_run1", "run-00"),
        ("localizer_run2", "run-07")
    ] + [("learningSession", f"run-{i:02d}") for i in range(1, 7)]

    # Get all MATLAB files
    matlab_files = glob.glob(os.path.join(args.matlab_dir, "LRN*_*.mat"))

    for run_type, run_id in run_order:
        if "learningSession" in run_type:
            # For "learningSession" runs, set the run_id directly
            learningSession_counter = int(run_id.split("-")[1])
        else:
            learningSession_counter = None  # Reset the counter for non-learningSession runs

        # Find the file that matches the current run_type
        matching_files = [f for f in matlab_files if run_type in f]
        
        if matching_files:
            matlab_file_path = matching_files[0]  # Process the first matching file
            matlab_files.remove(matlab_file_path)  # Ensure each file is processed only once
            
            filename = os.path.basename(matlab_file_path).rstrip('.mat')
            filename_parts = filename.split('_')
            subject_id = filename_parts[0]

            if learningSession_counter is not None:
                run_id = f"run-{learningSession_counter:02d}"
                learningSession_counter += 1
            
            trial_events, block_data = load_matlab_data(matlab_file_path)
            formatted_data = format_data_for_bids(trial_events, block_data)
            
            output_path = os.path.join(args.bids_root_dir, f"sub-{subject_id}", 'ses-2', 'func',
                                       f"sub-{subject_id}_ses-2_task-learn_{run_id}_events.tsv")
            
            save_as_tsv(formatted_data, output_path)
            logging.info(f"Saved .tsv file to: {output_path}")




