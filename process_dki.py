import os
import argparse
import subprocess
import shutil
import tempfile
import json
import pydicom
import glob

# <dicom_dir> <bids_root>

def extract_b0_field_identifier(dicom_file_path):
    """
    Extracts the B0FieldIdentifier from DICOM tags.
    
    Parameters:
    dicom_file_path (str): Path to the DICOM file.
    
    Returns:
    str: The SequenceName from the DICOM file.
    """
    dicom_data = pydicom.dcmread(dicom_file_path)
    return dicom_data.SequenceName

def extract_ids(dicom_dir):
    """
    Extracts the subject and session IDs from the DICOM directory path.
    
    Parameters:
    dicom_dir (str): Path to the DICOM directory.
    
    Returns:
    tuple: Subject ID and session ID as strings.
    """
    parts = dicom_dir.split('/')
    subject_id = next((part for part in parts if part.startswith('sub-')), None)
    session_id = next((part for part in parts if part.startswith('ses-')), None)
    return subject_id, session_id

def update_json_file(json_filepath, dicom_file_path, file_type, intended_for=None):
    """
    Updates specific fields in a JSON file based on DICOM data and file type.
    
    Parameters:
    json_filepath (str): Path to the JSON file to be updated.
    dicom_file_path (str): Path to the corresponding DICOM file.
    file_type (str): Type of file ('topup' or 'dwi').
    intended_for (str, optional): Path to the file that the data is intended for.
    """
    with open(json_filepath, 'r+') as file:
        data = json.load(file)
        
        # Update fields based on file type
        if file_type == 'topup':
            data['B0FieldIdentifier'] = extract_b0_field_identifier(dicom_file_path)
            data['IntendedFor'] = intended_for
        
        elif file_type == 'dwi':
            data['B0FieldSource'] = extract_b0_field_identifier(dicom_file_path)
        
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

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

# main code execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DKI DICOM files.')
    parser.add_argument('dicom_root_dir', type=str, help='Root DICOM directory.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    args = parser.parse_args()
    subject_id, session_id = extract_ids(args.dicom_root_dir)
    
    # Processing directories
    dki_dirs = ["DKI_BIPOLAR_2.5mm_64dir_58slices", "DKI_BIPOLAR_2.5mm_64dir_58slices_TOP_UP_PA"]
    
    for dki_dir in dki_dirs:
        dicom_dir = os.path.join(args.dicom_root_dir, dki_dir)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            run_dcm2niix(dicom_dir, tmpdirname)
            
            dicom_files = sorted(glob.glob(os.path.join(dicom_dir, '*')))
            dicom_index = 0  # Track DICOM files
            
            for file in os.listdir(tmpdirname):
                file_path = os.path.join(tmpdirname, file)
                
                # Define output directories and new filenames
                if "TOP_UP_PA" in dki_dir:
                    output_bids_dir = os.path.join(args.bids_root, subject_id, session_id, 'fmap')
                    new_file_name = f"{subject_id}_{session_id}_acq-topup_dir-PA_epi{os.path.splitext(file)[-1]}"
                else:
                    output_bids_dir = os.path.join(args.bids_root, subject_id, session_id, 'dwi')
                    new_file_name = f"{subject_id}_{session_id}_dir-AP_dwi{os.path.splitext(file)[-1]}"
                
                # Move and rename files
                os.makedirs(output_bids_dir, exist_ok=True)
                shutil.move(file_path, os.path.join(output_bids_dir, new_file_name))
                
                # Update JSON files
                if new_file_name.endswith('.json') and dicom_index < len(dicom_files):
                    corresponding_dicom = dicom_files[dicom_index]
                    dicom_index += 1  # Move to the next DICOM file
                    
                    if "TOP_UP_PA" in dki_dir:
                        # need to verify that this BIDs URI is resolvable to map the top up to the dwi file correctly. 
                        # see for more information: https://bids-specification.readthedocs.io/en/stable/02-common-principles.html#bids-uri
                        intended_for = f"bids::{subject_id}/{session_id}/dwi/{subject_id}_{session_id}_dir-AP_dwi.nii"
                        update_json_file(os.path.join(output_bids_dir, new_file_name), corresponding_dicom, 'topup', intended_for)
                    else:
                        update_json_file(os.path.join(output_bids_dir, new_file_name), corresponding_dicom, 'dwi')
