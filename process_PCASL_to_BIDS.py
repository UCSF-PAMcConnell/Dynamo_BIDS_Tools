"""
process_PCASL_to_BIDS.py

Description:
This script processes Pseudocontinuous Arterial Spin Labeling (PCASL) DICOM files into NIfTI format following the Brain Imaging Data Structure (BIDS) conventions. 
It includes functionalities for DICOM to NIfTI conversion using dcm2niix and additional BIDS-compliant metadata processing with cubids. The script checks for the installation of dcm2niix, pydeface, and cubids 
before executing relevant commands. It also handles file renaming.

Usage:
python process_PCASL_to_BIDS.py <dicom_root_dir> <bids_root_dir>
e.g.,
python process_PCASL_to_BIDS.py /path/to/dicom_root_dir /path/to/bids_root_dir 

Author: PAMcConnell
Created on: 20231112
Last Modified: 20231112

License: MIT License

Dependencies:
- Python 3.12
- dcm2niix (command-line tool) https://github.com/rordenlab/dcm2niix
- CuBIDS (command-line tool) https://cubids.readthedocs.io/en/latest/index.html
- os, shutil, subprocess, argparse, re (standard Python libraries)

Environment Setup:
- Ensure Python 3.12, dcm2niix and cubids are installed in your environment.
- You can install dcm2niix using conda: `conda install -c conda-forge dcm2niix`.
- Install cubids following the instructions provided in its documentation.
- Try 'pip install cubids'
- To set up the required environment, you can use the provided environment.yml file with Conda. <datalad.yml>

Change Log:
- 20231111: Initial version

*** NOTE: This script is still in development and may not work as expected regarding perfusion metadata.
# I'm not 100% on the ASL metadata defined below, needs verification - PAMcConnell 20231020
# https://github.com/npnl/ASL-Processing-Tips - updating accordingly based on this analysis - PAMcConnell20231026
# see also https://crnl.readthedocs.io/asl/index.html
# https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/glossary.html#term-Dwell-Time
#


"""

import os                     # Used for operating system dependent functionalities like file path manipulation.
import shutil                 # Provides high-level file operations like file copying and removal.
import logging                # Logging library, for tracking events that happen when running the software.
import argparse               # Parser for command-line options, arguments, and sub-commands.
import re                     # Regular expressions, useful for text matching and manipulation.
import subprocess             # Spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
import sys                    # Provides access to some variables used or maintained by the interpreter.
import json                   # JSON encoder and decoder.
import glob                   # Unix style pathname pattern expansion.
import pydicom                # Read, modify, and write DICOM files with Python.

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

    return log_file_path

# Checks if PCASL NIfTI files already exist in the specified BIDS output directory.
def check_existing_nifti(output_dir, subject_id, session_id):
    """
    Parameters:
    - output_dir (str): The BIDS output directory where NIfTI files are stored.
    - subject_id (str): The subject ID.
    - session_id (str): The session ID.

    Returns:
    - bool: True if T1w NIfTI files exist, False otherwise.
    """
    expected_nifti_file = os.path.join(output_dir, 'perf', f'{subject_id}_{session_id}_asl.nii')
    if os.path.isfile(expected_nifti_file):
        logging.info(f"T1-weighted NIfTI file already exists: {expected_nifti_file}")
        return True
    else:
        return False

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

# Reads and returns DICOM headers from the specified directory.
def read_dicom_headers(dicom_dir):
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f)) and f.startswith('MR.')]
    dicom_headers = [pydicom.dcmread(df, force=True) for df in dicom_files]
    return dicom_headers

# Calculates the number of volumes based on the DICOM headers.
def get_num_volumes(dicom_headers):
    return len(dicom_headers)

# Creates aslcontext.tsv file necessary for BIDS specification.
def create_aslcontext_file(num_volumes, output_dirs, subject_id, session_id):
    for output_dir in output_dirs:
        output_dir_perf = os.path.join(output_dir, 'perf')
        os.makedirs(output_dir_perf, exist_ok=True)

    asl_context_filepath = os.path.join(output_dir_perf, f'{subject_id}_{session_id}_aslcontext.tsv')
    with open(asl_context_filepath, 'w') as file:
        file.write('volume_type\n')
        for i in range(num_volumes):
            if i == 0:
                file.write('m0scan\n')
            else:
                file.write('control\n' if (i-1) % 2 == 0 else 'label\n')

# Updates the JSON sidecar file with necessary fields for BIDS compliance.
def update_json_file(json_filepath):
    with open(json_filepath, 'r+') as file:
        data = json.load(file)
        data['LabelingDuration'] = 0.7 # Bolus Duration
        # 82 RF blocks * 0.0185s RF Block Duration = 1.517 second "LabelTime"
        data['PostLabelingDelay'] = 1.000
        data['BackgroundSuppression'] = False
        data['M0Type'] = "Included"
        data['TotalAcquiredPairs'] = 6
        data['VascularCrushing'] = False
        # EffectiveEchoSpacing set to equal DwellTime 
        # (https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/glossary.html#term-Dwell-Time)
        # Need to verify if this needs to be adjusted based on SENSE parallel imaging parameters
        data['EffectiveEchoSpacing'] = 0.0000104
        data['B0FieldSource'] = "*fm2d2r"
        # Change Source 1 -> Source 2 to use spin-echo reverse phase field maps instead of default gre. 
        data['B0FieldSource2']=	"*epse2d1_104"
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

# Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format.
def run_dcm2niix(input_dir, output_dir, subject_id, session_id, log_file_path):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - output_dir (str): Directory where the converted NIfTI files will be saved.
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    This function uses the dcm2niix tool to convert DICOM files into NIfTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. The function assumes that dcm2niix is installed
    and accessible in the system's PATH.

    Usage Example:
    run_dcm2niix('/path/to/dicom', '/path/to/nifti', 'sub-01', 'ses-01')

    """
    try:
        # Ensure the output directory for anatomical scans exists.
        output_dir_anat = os.path.join(output_dir, 'perf')
        os.makedirs(output_dir_anat, exist_ok=True)
        cmd = [
            'dcm2niix',
            '-v', 'y', # Print verbose output.
            '-f', f'{subject_id}_{session_id}_asl', # Naming convention. 
            '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
            '-b', 'y', # Save BIDS metadata to .json sidecar. 
            '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
            '-x', 'n', # Do notCrop images. This will attempt to remove excess neck from 3D acquisitions.
            '-z', 'n', # Do not compress files.
            '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
            '-i', 'n', # Do not ignore derived, localizer and 2D images. 
            '-m', '2', # Merge slices from same series automatically based on modality. 
            '-o', output_dir_anat,
            input_dir
        ]
        
        # Execute dcm2niix with verbose output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Write verbose output to log file
        with open(log_file_path, 'a') as log_file:
            log_file.write("dcm2niix verbose output:\n")
            log_file.write(result.stdout if result.stdout else "")
            log_file.write(result.stderr if result.stderr else "")

        # Log conversion success.
        logging.info("dcm2niix conversion completed successfully.")
    
    # Log conversion errors.
    except subprocess.CalledProcessError as e:
        logging.error("dcm2niix conversion failed: %s", e)
        raise
    
    # Log other errors.
    except Exception as e:
        logging.error("An error occurred during dcm2niix conversion: %s", e)
        raise

# Executes the cubids-add-nifti-info command to add nifti header information to the BIDS dataset.
def run_cubids_add_nifti_info(bids_root_dir):
    """
    Executes the cubids-add-nifti-info command and logs changes made to .json sidecar files.
    
    Parameters:
    - bids_root_dir (str): Path to the BIDS dataset directory.
    """
    
    # Store the original contents of JSON files.
    json_files = glob.glob(os.path.join(bids_root_dir, '**', '*.json'), recursive=True)
    original_contents = {file: read_json_file(file) for file in json_files}

    try:
        cubids_add_nii_hdr_command = ['cubids-add-nifti-info', bids_root_dir]
        logging.info(f"Executing add nifti info: {cubids_add_nii_hdr_command}")
        
        # Run the command and capture output.
        result = subprocess.run(cubids_add_nii_hdr_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Log the standard output and error.
        logging.info("cubids-add-nifti-info output:\n%s", result.stdout)
        if result.stderr:
            logging.error("cubids-add-nifti-info error output:\n%s", result.stderr)
        logging.info("cubids-add-nifti-info executed successfully.")
        
        # Check and log changes in JSON files.
        for file in json_files:
            new_content = read_json_file(file)
            log_json_differences(original_contents[file], new_content)

        # Log successful execution of the command.
        logging.info("cubids-add-nifti-info executed and changes logged successfully.")


    # Catch cubids execution errors.
    except subprocess.CalledProcessError as e:
        logging.error(f"cubids-add-nifti-info execution failed: {e}")
        raise

# Executes the cubids-remove-metadata-fields command to remove metadata fields from the BIDS dataset.
def run_cubids_remove_metadata_fields(bids_root_dir, fields):
    """
    Executes the cubids-remove-metadata-fields command on the specified BIDS directory.
    
    Parameters:
    - bids_root_dir (str): Path to the BIDS dataset directory.
    - fields (list of str): List of metadata fields to remove.
    """
    fields_args = ['--fields', ','.join(fields)]  # Joining fields with commas for the command argument.
    try:
        cubids_remove_fields_command = ['cubids-remove-metadata-fields', bids_root_dir] + fields_args
        logging.info(f"Executing remove metadata fields: {cubids_remove_fields_command}")
        
        # Run the command and capture output.
        result = subprocess.run(cubids_remove_fields_command, check=True, stdout=subprocess.PIPE, text=True)

        # Log the standard output.
        logging.info("cubids-remove-metadata-fields output:\n%s", result.stdout)
        
        # Log successful execution.
        logging.info("cubids-remove-metadata-fields executed successfully.")
    
    # Catch cubids execution errors.
    except subprocess.CalledProcessError as e:
        logging.error(f"cubids-remove-metadata-fields execution failed: {e}")
        raise

# Check if dcm2niix is installed and accessible in the system's PATH.
def check_dcm2niix_installed():
    """
    Returns:
    bool: True if dcm2niix is installed, False otherwise.
    """
    dcm2niix_version_output = subprocess.getoutput('dcm2niix --version')
    if 'version' in dcm2niix_version_output:
        logging.info(f"dcm2niix is installed: {dcm2niix_version_output.splitlines()[0]}")
        return True
    else:
        logging.warning("dcm2niix is not installed.")
        return False
    
# Reads and returns the contents of a JSON file.
def read_json_file(filepath):
    """
    Parameters:
    - filepath (str): The full path to the JSON file.

    Returns:
    - dict: Contents of the JSON file.

    This function opens and reads a JSON file, then returns its contents as a Python dictionary.
    It handles file reading errors and logs any issues encountered while opening or reading the file.

    Usage Example:
    json_content = read_json_file('/path/to/file.json')
    """
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error reading JSON file {filepath}: {e}")
        raise

# Logs the differences between two JSON objects.
def log_json_differences(before, after):
    """
    Parameters:
    - before (dict): JSON object representing the state before changes.
    - after (dict): JSON object representing the state after changes.

    This function compares two JSON objects and logs any differences found.
    It's useful for tracking changes made to JSON files, such as metadata updates.

    The function assumes that 'before' and 'after' are dictionaries representing JSON objects.
    Differences are logged as INFO with the key and the changed values.

    Usage Example:
    log_json_differences(original_json, updated_json)
    """

    for key, after_value in after.items():
            before_value = before.get(key)
            if before_value != after_value:
                logging.info(f"Changed {key}: from '{before_value}' to '{after_value}'")
            elif key not in before:
                logging.info(f"Added {key}: {after_value}")

# Checks if cubids is installed and accessible in the system's PATH.
def check_cubids_installed():
    """
    Returns:
    bool: True if cubids is installed, False otherwise.
    """
    cubids_version_output = subprocess.getoutput('cubids-add-nifti-info -h')
    if 'cubids-add-nifti-info' in cubids_version_output:
        logging.info(f"cubids is installed: {cubids_version_output.splitlines()[0]}")
        return True
    else:
        logging.warning("cubids is not installed.")
        return False
   
# Main function for orchestrating the conversion process.
def main(dicom_root_dir, bids_root_dir):
    """
    Process DICOM files for ASL context and convert to NIfTI.

    Parameters:
    - dicom_root_dir (str): Root directory containing the DICOM directories.
    - bids_root (str): Root directory of the BIDS dataset.

    This function uses the dcm2niix tool to convert DICOM files into NIfTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. The function assumes that dcm2niix is installed
    and accessible in the system's PATH.
    """
    try:
        # Setup logging after extracting subject_id and session_id
        subject_id, session_id = extract_subject_session(dicom_root_dir)
        log_file_path = setup_logging(subject_id, session_id, bids_root_dir)
        logging.info(f"Processing subject: {subject_id}, session: {session_id}")

        # Specify the exact directory where the DICOM files are located
        dicom_dir = os.path.join(dicom_root_dir, 'tgse_pcasl_ve11c_from_USC')

        # Specify the exact directory where the NIfTI files will be saved
        output_dir = os.path.join(bids_root_dir, f'{subject_id}', f'{session_id}')

        # Check if PCASL NIfTI files already exist
        if not check_existing_nifti(output_dir, subject_id, session_id):
            if check_dcm2niix_installed():
                # Run dcm2niix for DICOM to NIfTI conversion.
                run_dcm2niix(dicom_dir, output_dir, subject_id, session_id, log_file_path)
            else:
                logging.error("dcm2niix is not installed. Cannot proceed with DICOM to NIfTI conversion.")
                return  # Exit the function if dcm2niix is not installed
            
        # Check if cubids is installed
        if check_cubids_installed():
            # Run cubids commands
            run_cubids_add_nifti_info(bids_root_dir)
            run_cubids_remove_metadata_fields(bids_root_dir, ['PatientBirthDate'])
        else:
            logging.warning("cubids is not installed. Skipping cubids commands.")
    
        # Reading DICOM headers
        dicom_headers = read_dicom_headers(dicom_dir)
        num_volumes = get_num_volumes(dicom_headers)

        # Check if the number of volumes is as expected
        if num_volumes != 12:
            logging.warning(f"Warning: Expected 12 volumes but found {num_volumes} volumes.")

        # Creating aslcontext.tsv, converting DICOM to NIfTI, and updating JSON files
        create_aslcontext_file(num_volumes, output_dir, subject_id, session_id)

    # Log other errors. 
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        raise

# Executes the main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files for ASL context and convert to NIfTI.')
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    parser.add_argument('bids_root_dir', type=str, help='Root directory of the BIDS dataset.')

    args = parser.parse_args()

    main(args.dicom_root_dir, args.bids_root_dir)