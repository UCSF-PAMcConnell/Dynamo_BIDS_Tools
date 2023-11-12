"""
process_T1_to_BIDS.py

Description:
This script processes T1-weighted DICOM files into NIfTI format following the Brain Imaging Data Structure (BIDS) conventions. 
It includes functionalities for DICOM to NIfTI conversion using dcm2niix, defacing NIfTI images with pydeface for enhanced privacy, 
and additional BIDS-compliant metadata processing with cubids. The script checks for the installation of dcm2niix, pydeface, and cubids 
before executing relevant commands. It also handles file renaming and applies defacing for anonymization.

Usage:
python process_T1_to_BIDS.py <dicom_root_dir> <bids_root_dir> [--pydeface]
e.g.,
python process_T1_to_BIDS.py /path/to/dicom_root_dir /path/to/bids_root_dir --pydeface

Author: PAMcConnell
Created on: 20231111
Last Modified: 20231112

License: MIT License

Dependencies:
- Python 3.12
- dcm2niix (command-line tool) https://github.com/rordenlab/dcm2niix
- pydeface (command-line tool) https://github.com/poldracklab/pydeface
- CuBIDS (command-line tool) https://cubids.readthedocs.io/en/latest/index.html
- logging, os, tempfile, shutil, glob, subprocess, argparse, sys, re, json (standard Python libraries)

Environment Setup:
- Ensure Python 3.12 is installed in your environment.
- Install dcm2niix and pydeface command-line tools. For dcm2niix, use `conda install -c conda-forge dcm2niix`.
- Pydeface installation may vary depending on your system. Refer to the official pydeface documentation for instructions.
- try `conda install -c conda-forge pydeface`
- Install cubids following the instructions provided in its documentation.
- Try 'pip install cubids'
- To set up the required environment, use the provided environment.yml file with Conda. <datalad.yml>

Change Log:
- 20231111: Initial version.
- 20231112: added verbose dcm2niix logging. Standardized output file path. 
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
import tempfile               # Generate temporary files and directories.

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

# Checks if T1-weighted NIfTI files already exist in the specified BIDS output directory.
def check_existing_nifti(output_dir_anat, subject_id, session_id):
    """
    Parameters:
    - output_dir_anat (str): The BIDS output directory where NIfTI files are stored.
    - subject_id (str): The subject ID.
    - session_id (str): The session ID.

    Returns:
    - bool: True if T1w NIfTI files exist, False otherwise.
    """
    expected_nifti_file = os.path.join(output_dir_anat, f'{subject_id}_{session_id}_T1w.nii')
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

# Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format.
def run_dcm2niix(input_dir, output_dir_anat, subject_id, session_id, log_file_path):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - output_dir_anat (str): Directory where the converted NIfTI files will be saved.
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    This function uses the dcm2niix tool to convert DICOM files into NIfTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. The function assumes that dcm2niix is installed
    and accessible in the system's PATH.

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<anat_dicoms>', '/bids_root_dir/sub-01/ses-01/anat', 'sub-01', 'sub-01', 'ses-01')

    """
    try:
        # Ensure the output directory for anatomical scans exists.
        os.makedirs(output_dir_anat, exist_ok=True)
        base_cmd = [
            'dcm2niix',
            '-f', f'{subject_id}_{session_id}_T1w', # Naming convention. 
            '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
            '-b', 'y', # Save BIDS metadata to .json sidecar. 
            '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
            '-x', 'y', # Crop images. This will attempt to remove excess neck from 3D acquisitions.
            '-z', 'n', # Do not compress files.
            '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
            '-i', 'n', # Do not ignore derived, localizer and 2D images. 
            '-m', '2', # Merge slices from same series automatically based on modality. 
            '-o', output_dir_anat,
            input_dir
        ]
        
        # Run the actual conversion without verbose output.
        subprocess.run(base_cmd) #capture_output=False, text=False)
        logging.info(f"dcm2niix conversion completed successfully to {output_dir_anat}.")

    # Log conversion errors.
    except subprocess.CalledProcessError as e:
        logging.error("dcm2niix conversion failed: %s", e)
        raise
    
    # Log other errors.
    except Exception as e:
        logging.error("An error occurred during dcm2niix conversion: %s", e)
        raise

# Runs the dcm2niix conversion tool to produce verbose output to logfile. 
def run_dcm2niix_verbose(input_dir, temp_dir, subject_id, session_id, log_file_path):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - temp_dir (str): Directory where the converted NIfTI files will be saved and deleted. 
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    The function logs verbose output to the specified log file. 

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<perf_dicoms>, 'temp_dir', 'sub-01', 'ses-01')

    """
    try:
        verbose_cmd = [
        'dcm2niix',
        '-f', f'{subject_id}_{session_id}_T1w', # Naming convention. 
        '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
        '-b', 'y', # Save BIDS metadata to .json sidecar. 
        '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
        '-x', 'n', # Do notCrop images. This will attempt to remove excess neck from 3D acquisitions.
        '-z', 'n', # Do not compress files.
        '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
        '-i', 'n', # Do not ignore derived, localizer and 2D images. 
        '-m', '2', # Merge slices from same series automatically based on modality. 
        '-v', 'y', # Print verbose output to logfile.
        '-o', temp_dir,
        input_dir
    ]
        
        # Create a temporary directory for the verbose output run.
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(log_file_path, 'a') as log_file:
                result = subprocess.run(verbose_cmd, check=True, stdout=log_file, stderr=log_file)
                logging.info(result.stdout)
                if result.stderr:
                    logging.error(result.stderr)

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

# Executes the pydeface command to anonymize NIfTI images by removing facial features.
def run_pydeface(output_dir_anat, subject_id, session_id):
    """
    Parameters:
    - output_dir_anat (str): Directory where the NIfTI files are stored.
    - subject_id (str): Subject ID used in the BIDS file naming.
    - session_id (str): Session ID used in the BIDS file naming.

    The function constructs a pydeface command to deface the anatomical MRI images (T1-weighted images)
    for the specified subject and session. The defaced image is saved with the same filename, 
    overwriting the original by using the '--force' flag.

    Pydeface is used to enhance the privacy and anonymity of the MRI data by removing facial characteristics 
    that could potentially identify subjects. This is especially important in shared datasets.

    Usage Example:
    run_pydeface('/path/to/nifti/output', 'sub-01', 'ses-01')

    Dependencies:
    - subprocess module for executing shell commands.
    - logging module for logging operations.

    """
    try:
        # Construct the pydeface command to overwrite the original T1w image.
        pydeface_command = [
            'pydeface',
            f"{output_dir_anat}/{subject_id}_{session_id}_T1w.nii",
            '--outfile',
            f"{output_dir_anat}/{subject_id}_{session_id}_T1w.nii",
            '--force'
        ]
        logging.info(f"Executing pydeface: {' '.join(pydeface_command)}")
        result = subprocess.run(pydeface_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Log the standard output and error.
        logging.info("pydeface output:\n%s", result.stdout)
        if result.stderr:
            logging.error("pydeface error output:\n%s", result.stderr)
        logging.info("pydeface executed successfully.")

    except subprocess.CalledProcessError as e:
        logging.error("pydeface execution failed: %s", e)
        raise
    except Exception as e:
        logging.error("An error occurred during pydeface execution: %s", e)
        raise

# Renames the cropped NIfTI file to overwrite the original T1w file in BIDS format.
def rename_cropped_file(output_dir_anat, subject_id, session_id):
    """
    This function is particularly useful in workflows where post-processing steps 
    (like cropping) generate new files that should replace the original ones.

    Parameters:
    - output_dir_anat (str): Directory where the conversion results are saved.
    - subject_id (str): Subject ID, typically in the format 'sub-XX'.
    - session_id (str): Session ID, typically in the format 'ses-XX'.

    The function assumes the presence of a cropped file named 'sub-XX_ses-XX_T1w_Crop_1.nii'.
    If this file exists, it is renamed to overwrite 'sub-XX_ses-XX_T1w.nii', ensuring 
    consistency with BIDS naming conventions.

    Usage Example:
    rename_cropped_file('/path/to/nifti/output', 'sub-01', 'ses-01')
    """
    
    try:
        cropped_file_path = os.path.join(output_dir_anat, f'{subject_id}_{session_id}_T1w_Crop_1.nii')
        original_file_path = os.path.join(output_dir_anat, f'{subject_id}_{session_id}_T1w.nii')
        
        if os.path.exists(cropped_file_path):
            shutil.move(cropped_file_path, original_file_path)
            logging.info(f"Cropped file renamed to overwrite the original T1w file: {original_file_path}")
        else:
            logging.warning(f"No cropped file found to rename: {cropped_file_path}")
    
    # Log errors.
    except Exception as e:
            logging.error(f"Error occurred while renaming cropped file: {e}")
            raise
    
# Checks if pydeface is installed and accessible in the system's PATH.
def check_pydeface_installed():
    """
    Checks if pydeface is installed and accessible in the system's PATH.

    Returns:
    bool: True if pydeface is installed, False otherwise.
    """
    pydeface_version_output = subprocess.getoutput('pydeface --version')
    if 'pydeface 2.0.2' in pydeface_version_output: # change version number as needed.
        logging.info(f"pydeface is installed: {pydeface_version_output.splitlines()[0]}")
        return True
    else:
        logging.warning("pydeface version 2.02 is not installed.")
        return False
    
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
    
# The main function to convert T1w DICOM files to NIfTI format within a BIDS compliant structure.
def main(dicom_root_dir, bids_root_dir, run_pydeface_func=False):
    """
    This function sets up logging, executes the DICOM to NIfTI conversion using dcm2niix, renames any cropped files, 
    and optionally runs pydeface for anonymization.

    Parameters:
    - dicom_root_dir (str): The root directory containing the DICOM files.
    - bids_root_dir (str): The root directory for the BIDS dataset where the NIfTI files will be saved.

    The function assumes that 'pydeface' and 'dcm2niix' are installed and accessible in the system's PATH.

    Usage Example:
    main('/path/to/dicom_root_dir', '/path/to/bids_root_dir')

    Dependencies:
    - dcm2niix for DICOM to NIfTI conversion.
    - pydeface for defacing NIfTI images.
    """
    try:
        # Setup logging after extracting subject_id and session_id.
        subject_id, session_id = extract_subject_session(dicom_root_dir)
        log_file_path = setup_logging(subject_id, session_id, bids_root_dir)
        logging.info(f"Processing subject: {subject_id}, session: {session_id}")

        # Specify the exact directory where the DICOM files are located.
        dicom_dir = os.path.join(dicom_root_dir, 't1_mprage_sag_p2_iso')

        # Specify the exact directory where the NIfTI files will be saved.
        output_dir_anat = os.path.join(bids_root_dir, f'{subject_id}', f'{session_id}', 'anat')

        # Check if T1w NIfTI files already exist.
        if not check_existing_nifti(output_dir_anat, subject_id, session_id):
            if check_dcm2niix_installed():
                # Run dcm2niix for DICOM to NIfTI conversion.
                run_dcm2niix(dicom_dir, output_dir_anat, subject_id, session_id, log_file_path)
            else:
                logging.error("dcm2niix is not installed. Cannot proceed with DICOM to NIfTI conversion.")
                return  # Exit the function if dcm2niix is not installed.
            
            # Rename any cropped files if they exist and overwrite the original T1-weighted NIfTI file.
            rename_cropped_file(output_dir_anat, subject_id, session_id)
        else:
            logging.info("Skipping DICOM to NIfTI conversion as T1-weighted NIfTI file already exists.")

        # Optional pydeface execution.
        if run_pydeface_func and check_pydeface_installed():
            run_pydeface(output_dir_anat, subject_id, session_id)

        # Log pydeface execution errors.
        elif run_pydeface:
            logging.warning("Skipping pydeface execution as it is not installed.")
        else:
            logging.info("Pydeface execution is not enabled.")

        # Check if cubids is installed.
        if check_cubids_installed():
            # Run cubids commands.
            run_cubids_add_nifti_info(bids_root_dir)
            run_cubids_remove_metadata_fields(bids_root_dir, ['PatientBirthDate'])
        else:
            logging.warning("cubids is not installed. Skipping cubids commands.")
        
        if check_dcm2niix_installed():
                # Run dcm2niix for verbose output.
                with tempfile.TemporaryDirectory() as temp_dir:
                   run_dcm2niix_verbose(dicom_dir, temp_dir, subject_id, session_id, log_file_path)
        else:
            logging.error("dcm2niix is not installed. Cannot proceed with DICOM to NIfTI conversion.")
            return  # Exit the function if dcm2niix is not installed.
     
    # Log other errors. 
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        raise

# Main code execution starts here.
if __name__ == "__main__":
    """
    Entry point of the script when executed from the command line.

    Parses command-line arguments to determine the directories for DICOM files and BIDS dataset,
    and an optional flag to run pydeface for defacing the images.

    Usage:
    process_T1_to_BIDS.py <dicom_root_dir> <bids_root> [--pydeface]
    """
    
    # Set up an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description='Process DICOM files and convert them to NIfTI format following BIDS conventions.')
    
    # Add arguments to the parser.
    
    # The first argument is the root directory containing the DICOM directories.
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    
    # The second argument is the root directory of the BIDS dataset.
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    # The third (optional) argument specifies whether to run pydeface for defacing the images.
    parser.add_argument('--pydeface', action='store_true', help='Optionally run pydeface for defacing the images.')
    
    # Parse the arguments provided by the user.
    args = parser.parse_args()
    
    # Run the main function with the parsed arguments.
    main(args.dicom_root_dir, args.bids_root, args.pydeface)