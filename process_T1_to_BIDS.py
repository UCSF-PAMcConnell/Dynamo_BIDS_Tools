"""
process_T1_to_BIDS.py

Description:
This script processes DICOM files into NIfTI format following the Brain Imaging Data Structure (BIDS) conventions. 
It also handles file renaming and applies defacing for anonymization. The script uses dcm2niix for conversion and pydeface for defacing.

Usage:
python BIDS_process_dicom_to_nifti.py <dicom_root_dir> <bids_root>
e.g.,
/dataset_root_dir/dicom_root_dir/subject_folder # <dicom_root_dir>
/dataset_root_dir/bids_root_dir/ # <bids_root>

Author: PAMcConnell
Created on: 20231111
Last Modified: 20231111

License: MIT License

Dependencies:
- Python 3.12
- dcm2niix
- pydeface
- os, shutil, subprocess, argparse, re (standard Python libraries)

Environment Setup:
- Ensure Python 3.12, dcm2niix, and pydeface are installed in your environment.
- You can install using conda: `conda install -c conda-forge pydicom dcm2niix.
- To set up the required environment, you can use the provided environment.yml file with Conda. <datalad.yml>

Change Log:
- 20231111: Initial version
"""

import os                     # Used for operating system dependent functionalities like file path manipulation.
import shutil                 # Provides high-level file operations like file copying and removal.
import logging                # Logging library, for tracking events that happen when running the software.
import argparse               # Parser for command-line options, arguments, and sub-commands.
import re                     # Regular expressions, useful for text matching and manipulation.
import subprocess             # Spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
import sys                    # Provides access to some variables used or maintained by the interpreter.
import dcm2niix               # Tool for converting DICOM files to NIfTI format.    
import pydeface               # Tool for defacing images.

# Set up logging for individual archive logs.
def setup_logging(subject_id, session_id, bids_root_dir):
    """
    Sets up logging for the script, creating log files in a specified directory.

    Parameters:
    - subject_id (str): The identifier for the subject.
    - session_id (str): The identifier for the session.
    - bids_root_dir (str): The root directory of the BIDS dataset.

    This function sets up a logging system that writes logs to both a file and the console. 
    The log file is named based on the subject ID, session ID, and the script name. 
    It's stored in a 'logs' directory within the 'doc' folder by subject ID, which is located at the same 
    level as the BIDS root directory.

    The logging level is set to INFO, meaning it captures all informational, warning, and error messages.

    Usage Example:
    setup_logging('sub-01', 'ses-1', '/path/to/bids_root_dir')
    """

    # Extract the base name of the script without the .py extension.
    script_name = os.path.basename(__file__).replace('.py', '')

    # Construct the log directory path within 'doc/logs'
    log_dir = os.path.join(os.path.dirname(bids_root_dir), 'doc', 'logs', script_name, subject_id)

    # Create the log directory if it doesn't exist.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the log file name using subject ID, session ID, and script name.
    log_file_name = f"{subject_id}_{session_id}_{script_name}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    # Configure file logging.
    logging.basicConfig(
        level=logging.INFO,
        # filename='process_physio_ses_2.log', # Uncomment this line to save log in script execution folder.
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w' # 'w' mode overwrites existing log file.
    )

    # If you also want to log to console.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Logging setup complete. Log file: {log_file_path}")

# Extract the subject and session IDs from the provided physio root directory path.
def extract_subject_session(dicom_root_dir):
    """
    Parameters:
    - dicom_root_dir (str): The directory path that includes subject and session information. 
                             This path should follow the BIDS convention, containing 'sub-' and 'ses-' prefixes.

    Returns:
    - subject_id (str): The extracted subject ID.
    - session_id (str): The extracted session ID.

    Raises:
    - ValueError: If the subject_id and session_id cannot be extracted from the path.

    This function assumes that the directory path follows the Brain Imaging Data Structure (BIDS) naming convention. 
    It uses regular expressions to find and extract the subject and session IDs from the path.

    Usage Example:
    subject_id, session_id = extract_subject_session('/path/to/data/sub-01/ses-1/dicom_sorted')

    Note: This function will raise an error if it cannot find a pattern matching the BIDS convention in the path.
    """

    # Normalize the path to remove any trailing slashes for consistency.
    dicom_root_dir = os.path.normpath(dicom_root_dir)

    # The pattern looks for 'sub-' followed by any characters until a slash, and similar for 'ses-'.
    match = re.search(r'(sub-[^/]+)/(ses-[^/]+)', dicom_root_dir)
    if not match:
        raise ValueError("Unable to extract subject_id and session_id from path: %s", dicom_root_dir)
    
    subject_id, session_id = match.groups()

    return subject_id, session_id

# Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format and names the output files according to BIDS conventions.
def run_dcm2niix(input_dir, output_dir, subject_id, session_id):
    """
    Parameters:
    input_dir (str): Input directory containing DICOM files.
    output_dir (str): Directory where the conversion results will be saved.
    subject_id (str): Subject ID extracted from the DICOM directory path.
    session_id (str): Session ID extracted from the DICOM directory path.
    """
    #dcm2niix_path = os.path.expanduser('~/Documents/MATLAB/software/iNR/BIDS_tools/dcm2niix')
    output_dir_anat = os.path.join(output_dir, 'anat')
    os.makedirs(output_dir_anat, exist_ok=True)
    cmd = [
        dcm2niix,
        '-f', f'sub-{subject_id}_ses-{session_id}_T1w',
        'l', 'y',
        '-p', 'n',
        '-x', 'y',
        '-z', 'n',
        '-ba', 'n',
        '-o', output_dir_anat,
        input_dir
    ]
    subprocess.run(cmd)

# Renames the cropped file to overwrite the original T1w file.
def rename_cropped_file(output_dir, subject_id, session_id):
    """
    Renames the cropped file to overwrite the original T1w file.
    
    Parameters:
    output_dir (str): Directory where the conversion results are saved.
    subject_id (str): Subject ID extracted from the DICOM directory path.
    session_id (str): Session ID extracted from the DICOM directory path.
    """
    cropped_file_path = os.path.join(output_dir, 'anat', f'sub-{subject_id}_ses-{session_id}_T1w_Crop_1.nii')
    original_file_path = os.path.join(output_dir, 'anat', f'sub-{subject_id}_ses-{session_id}_T1w.nii')
    
    if os.path.exists(cropped_file_path):
        shutil.move(cropped_file_path, original_file_path)
        logging.info(f"Cropped file has been renamed to overwrite the original T1w file: {original_file_path}")

def main(dicom_root_dir, bids_root_dir):

    # Setup logging after extracting subject_id and session_id.
    subject_id, session_id = extract_subject_session(dicom_root_dir)
    setup_logging(subject_id, session_id, bids_root_dir)
    logging.info(f"Processing subject: {subject_id}, session: {session_id}")

    # Specify the exact directory where the DICOM files are located within the root directory
    dicom_root_dir = os.path.join(args.dicom_root_dir, 't1_mprage_sag_p2_iso')

    # Specify the exact directory where the NIfTI files will be saved within the root directory
    output_dir = os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}')
    
    run_dcm2niix(dicom_root_dir, output_dir, subject_id, session_id)
    rename_cropped_file(output_dir, subject_id, session_id)

    # Using the full path of pydeface to execute the command
    # pydeface_path = "~/anaconda3/envs/fmri/bin/pydeface"
    pydeface_command = f"python {pydeface} {output_dir}/anat/'sub-{subject_id}_ses-{session_id}_T1w'.nii --outfile {output_dir}/anat/'sub-{subject_id}_ses-{session_id}_T1w'.nii --force"
    
    logging.info(f"Executing: {pydeface_command}")
    
    # Uncomment the following line to actually execute the command
    subprocess.run(pydeface_command, shell=True)

# Main code execution starts here
if __name__ == "__main__":
    
    # Configure the argument parser
    parser = argparse.ArgumentParser(description='Process DICOM files and convert them to NIfTI format following BIDS conventions.')
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    args = parser.parse_args()
    
    # Run the main function
    main(args.dicom_root_dir, args.bids_root)