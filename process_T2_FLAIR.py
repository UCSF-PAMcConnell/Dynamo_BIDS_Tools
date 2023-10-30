import os
import argparse
import subprocess
import re

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

def run_dcm2niix(input_dir, output_dir, subject_id, session_id):
    """
    Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format and names the output files according to BIDS conventions.
    
    Parameters:
    input_dir (str): Input directory containing DICOM files.
    output_dir (str): Directory where the conversion results will be saved.
    subject_id (str): Subject ID extracted from the DICOM directory path.
    session_id (str): Session ID extracted from the DICOM directory path.
    """
    dcm2niix_path = os.path.expanduser('~/Documents/MATLAB/software/iNR/BIDS_tools/dcm2niix')
    output_dir_anat = os.path.join(output_dir, 'anat')
    os.makedirs(output_dir_anat, exist_ok=True)
    cmd = [
        dcm2niix_path,
        '-f', f'sub-{subject_id}_ses-{session_id}_FLAIR',
        'l', 'y',
        '-p', 'n',
        '-x', 'y',
        '-z', 'n',
        '-ba', 'n',
        '-o', output_dir_anat,
        input_dir
    ]
    subprocess.run(cmd)

# Main code execution starts here
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files and convert them to NIfTI format following BIDS conventions.')
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    args = parser.parse_args()
    
    # Specify the exact directory where the DICOM files are located within the root directory
    dicom_dir = os.path.join(args.dicom_root_dir, 't2_tse_dark-fluid_tra_3mm')

    #dicom_dir = args.dicom_root_dir  # Using the provided DICOM root directory directly
    
    subject_id, session_id = extract_ids(dicom_dir)
    
    output_dir = os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}')
    
    run_dcm2niix(dicom_dir, output_dir, subject_id, session_id)