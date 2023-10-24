import os
import argparse
import subprocess
import shutil
import tempfile
import json

# input arguments: <dicom_root_dir> <bids_root> <num_runs>

def update_json_file(json_filepath):
    """
    Updates specific fields in a JSON file. Specifically, this function
    adds the 'TaskName' field with the value 'rest'.
    
    Parameters:
    json_filepath (str): Path to the JSON file to be updated.
    """
    with open(json_filepath, 'r+') as file:
        data = json.load(file)
        data['TaskName'] = 'rest'
        data['B0FieldSource'] = "*fm2d2r"
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
    parts = dicom_dir.split('/')
    subject_id = next((part for part in parts if part.startswith('sub-')), None)
    session_id = next((part for part in parts if part.startswith('ses-')), None)
    return subject_id, session_id

def run_dcm2niix(input_dir, output_dir_temp):
    """
    Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format.
    
    Parameters:
    input_dir (str): Input directory containing DICOM files.
    output_dir_temp (str): Temporary directory where the conversion results will be saved.
    """
    cmd = [
        '/Users/PAM201/Documents/MATLAB/software/iNR/BIDS_tools/dcm2niix',
        '-f', '"%p_%s"',
        '-p', 'y',
        '-z', 'n',
        '-ba', 'n',
        '-o', output_dir_temp,
        input_dir
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files and convert to NIfTI in BIDS format.')
    #parser.add_argument('base_dicom_dir', type=str, help='Base DICOM directory.')
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    parser.add_argument('num_runs', type=int, help='Number of runs.')
    
    args = parser.parse_args()
    
    # Specify the exact directory where the DICOM files are located within the root directory
    base_dicom_dir = os.path.join(args.dicom_root_dir)

    # Extract subject and session IDs from the DICOM directory
    subject_id, session_id = extract_ids(args.dicom_root_dir)
    
    # Loop through each run, processing the DICOM files and organizing them in BIDS format
    for run in range(1, args.num_runs+1):
        for suffix in ['', '_SBref']:
            dicom_dir = os.path.join(base_dicom_dir, f'Resting_{run}{suffix}')
            
            # Create a temporary directory for the dcm2niix output
            with tempfile.TemporaryDirectory() as tmpdirname:
                run_dcm2niix(dicom_dir, tmpdirname)
                
                # Define the output directory based on BIDS structure
                func_dir_bids = os.path.join(args.bids_root, subject_id, session_id, 'func')
                os.makedirs(func_dir_bids, exist_ok=True)
                
                # Process and move each output file to the BIDS directory, applying naming conventions
                for old_file in os.listdir(tmpdirname):
                    old_filepath = os.path.join(tmpdirname, old_file)
                    
                    new_file = f"{subject_id}_{session_id}_task-rest_run-{run:02d}"
                    if suffix == '_SBref':
                        new_file += '_sbref'
                    else:
                        new_file += '_bold'
                    new_file += os.path.splitext(old_file)[-1]
                    
                    new_filepath = os.path.join(func_dir_bids, new_file)
                    shutil.move(old_filepath, new_filepath)
                    
                    # Update the JSON file with the task name
                    if new_filepath.endswith('.json'):
                        update_json_file(new_filepath)
