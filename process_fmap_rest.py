import os
import argparse
import subprocess
import re
import glob
import json
import shutil
import tempfile

# input arguments: <dicom_dir> <bids_root>
# Requires definition of "IntendedFor" task volumes in associated field maps
#{
#   "IntendedFor": [
#        "bids::sub-01/ses-pre/func/sub-01_ses-pre_task-motor_run-1_bold.nii.gz",
#        "bids::sub-01/ses-pre/func/sub-01_ses-pre_task-motor_run-2_bold.nii.gz"
#    ]
#}

def update_json_file(json_filepath):
    """
    Updates specific fields in a JSON file.
    
    Parameters:
    json_filepath (str): Path to the JSON file to be updated.
    """
    with open(json_filepath, 'r+') as file:
        data = json.load(file)
        data['TaskName'] = 'rest'
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

def extract_ids(dicom_dir):
    """
    Extracts subject and session IDs from the DICOM directory path.
    
    Parameters:
    dicom_dir (str): Path to the DICOM directory.
    
    Returns:
    tuple: Subject ID and session ID as strings.
    """
    match_subject = re.search(r'sub-(\w+)', dicom_dir)
    match_session = re.search(r'ses-(\w+)', dicom_dir)
    
    if match_subject and match_session:
        return match_subject.group(1), match_session.group(1)
    else:
        raise ValueError("Could not extract subject and session IDs from the directory path.")

def run_dcm2niix(input_dir, output_dir_temp):
    """
    Runs the dcm2niix conversion and saves the output in a temporary directory.
    
    Parameters:
    input_dir (str): Input directory containing DICOM files.
    output_dir_temp (str): Temporary directory where the conversion results will be saved.
    """
    cmd = [
        '/Users/PAM201/Documents/MATLAB/software/iNR/BIDS_tools/dcm2niix',
        '-f', '"sub-%i_%p"',
        '-p', 'y',
        '-z', 'n',
        '-ba', 'n',
        '-o', output_dir_temp,
        input_dir
    ]
    subprocess.run(cmd)

# Main code execution starts here when the script is run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files and convert to NIfTI.')
    parser.add_argument('dicom_dirs', type=str, nargs='+', help='Directories containing the DICOM files.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    args = parser.parse_args()
    
    run_number = 1  # Initialize the run number
    
    # Loop through each specified DICOM directory
    for i, dicom_dir in enumerate(args.dicom_dirs, start=1):
        subject_id, session_id = extract_ids(dicom_dir)
        
        # Check if the directory is for SBref or bold images
        is_sbref = 'SBref' in dicom_dir
        
        # Create a temporary directory to store the dcm2niix output
        with tempfile.TemporaryDirectory() as tmpdirname:
            run_dcm2niix(dicom_dir, tmpdirname)
            
            # Define the BIDS 'func' directory where the files should ultimately be saved
            func_dir_bids = os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}', 'func')
            os.makedirs(func_dir_bids, exist_ok=True)
            
            suffix = '_sbref' if is_sbref else '_bold'
            
            # Iterate over the files in the temporary directory, renaming and moving them as necessary
            for old_file in glob.glob(os.path.join(tmpdirname, '*.*')):
                # Determine the new filename based on whether the file is SBref or bold
                new_filename = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{run_number:02d}{suffix}"
                if old_file.endswith('.nii'):
                    new_filename += '.nii'
                elif old_file.endswith('.json'):
                    new_filename += '.json'
                
                new_file_path = os.path.join(func_dir_bids, new_filename)
                shutil.copy2(old_file, new_file_path)
                
                if old_file.endswith('.json'):
                    update_json_file(new_file_path)
            
            # Increment the run number only when processing bold images (not SBref)
            if not is_sbref:
                run_number += 1
