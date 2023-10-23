import os
import argparse
import subprocess
import re
import glob
import json
import shutil
import tempfile

def update_json_file(json_filepath):
    """
    Updates specific fields in a JSON file.
    
    Parameters:
    json_filepath (str): Path to the JSON file to be updated.
    """
    with open(json_filepath, 'r+') as file:
        data = json.load(file)
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
    parser.add_argument('dicom_dir', type=str, help='Directory containing the DICOM files.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    args = parser.parse_args()
    
    subject_id, session_id = extract_ids(args.dicom_dir)
    
    # Create a temporary directory to store the dcm2niix output
    with tempfile.TemporaryDirectory() as tmpdirname:
        run_dcm2niix(args.dicom_dir, tmpdirname)
        
        # Define the BIDS 'anat' directory where the files should ultimately be saved
        anat_dir_bids = os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}', 'anat')
        os.makedirs(anat_dir_bids, exist_ok=True)
        
        # Copy files from temporary directory to the BIDS 'anat' directory with renaming
        for ext in ['nii', 'json']:
            old_files = glob.glob(os.path.join(tmpdirname, f'*.{ext}'))
            for old_file in old_files:
                file_basename = os.path.basename(old_file)
                new_file = os.path.join(anat_dir_bids, f'sub-{subject_id}_ses-{session_id}_FLAIR.{ext}')
                shutil.copy2(old_file, new_file)
                
                if ext == 'json':
                    update_json_file(new_file)
