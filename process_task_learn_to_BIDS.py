"""
process_task_learn_to_BIDS.py

Description:

This script is designed to process Task fMRI DICOM files into NIfTI format in 
compliance with Brain Imaging Data Structure (BIDS) conventions. 
It encompasses DICOM to NIfTI conversion using the dcm2niix tool, 
along with the application of additional BIDS-compliant metadata processing using cubids. 
Before executing, the script verifies the installation of dcm2niix and cubids. 
It features robust error handling and detailed logging for each processing step, 
ensuring reliability and transparency in data conversion.

Usage:
Invoke the script from the command line with the following format:
python process_task_learn_to_BIDS.py <dicom_root_dir> <bids_root_dir> 

Example usage:
python process_task_learn_to_BIDS.py /path/to/dicom_root_dir /path/to/bids_root_dir

Author: PAMcConnell
Created on: 20231112
Last Modified: 20231112
Version: 1.0.0

License:
This software is released under the MIT License.

Dependencies:
- Python 3.12
- dcm2niix (command-line tool): Essential for converting DICOM to NIfTI format. (https://github.com/rordenlab/dcm2niix)
- CuBIDS (command-line tool): Utilized for handling BIDS-compliant metadata. (https://cubids.readthedocs.io/en/latest/index.html)
- Python standard libraries: logging, os, tempfile, shutil, glob, subprocess, argparse, sys, re, json.

Environment Setup:
- Ensure Python 3.12 is installed in your environment.
- Install dcm2niix and pydeface command-line tools. Use `conda install -c conda-forge dcm2niix` for dcm2niix installation.
- Install cubids as per the instructions in its documentation. Try using 'pip install cubids'.
- To establish the required environment, utilize the provided 'environment.yml' file with Conda.

Change Log:
- 20231111: Initial release of the script with basic functionality for DICOM to NIfTI conversion.
- 20231112: Enhanced functionality with verbose logging from dcm2niix, standardized output file paths, and improved error handling.
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

# Checks if Task fmri NIfTI files already exist in the specified BIDS output directory.
def check_existing_nifti(output_dir_func, subject_id, session_id):
    """
    Parameters:
    - output_dir_func (str): The BIDS output directory where NIfTI files are stored.
    - subject_id (str): The subject ID.
    - session_id (str): The session ID.

    Returns:
    - bool: True if Task fMRI NIfTI files exist, False otherwise.
    """
    expected_nifti_file = os.path.join(output_dir_func, f'{subject_id}_{session_id}_task-learn_run-01_bold.nii')
    if os.path.isfile(expected_nifti_file):
        print(f"Task fmri NIfTI file already exists: {expected_nifti_file}")
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

    print(f"Subject ID: %s, Session ID: %s", subject_id, session_id)
    return subject_id, session_id

# Updates the JSON sidecar file with specific fields required for BIDS compliance in TASK FMRI datasets.
def update_json_file(new_filepath):
    """
    Parameters:
    - json_filepath (str): Path to the JSON sidecar file.

    This function updates the specified JSON file with fields relevant to Task imaging.

    The function handles the reading and writing of the JSON file, ensuring that the file is properly updated
    and formatted.

    Usage Example:
    update_json_file('/path/to/sidecar.json')

    Dependencies:
    - json module for reading and writing JSON files.
    - os and sys modules for file operations and system-level functionalities.

    *** Note: Refer to MR protocol documentation in /doc/MR_protocols for more information. ***
    """
    try:
        with open(new_filepath, 'r+') as file:
            data = json.load(file)
            logging.info(f"Original data: {data}")

            # Update with specific Task metadata
            data['TaskName'] = 'learn'
            logging.info(f"Task Name: {data['TaskName']}")
            data['B0FieldSource'] = "*fm2d2r"
            logging.info(f"B0 Field Source: {data['B0FieldSource']}")
            data['B0FieldSource2']=	"*epse2d1_104"
            logging.info(f"B0 Field Source 2: {data['B0FieldSource2']}")
            
            # Write back the updated data and truncate the file to the new data length.
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()
            logging.info(f"Updated data: {data}")
        
        logging.info(f"Updated JSON file at {new_filepath} with TASK FMRI-specific metadata.")
    
    # Catch issues with reading or writing to the JSON file.
    except IOError as e:
        logging.error(f"Error reading or writing to JSON file at {new_filepath}. Error: {e}")
        raise

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON data in file at {new_filepath}. Error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred while updating JSON file at {new_filepath}. Error: {e}")
        raise

# Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format.
def run_dcm2niix(input_dir, temp_dir, subject_id, session_id):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - output_dir_func (str): Directory where the converted NIfTI files will be saved.
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    This function uses the dcm2niix tool to convert DICOM files into NIfTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. The function assumes that dcm2niix is installed
    and accessible in the system's PATH.

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<func_dicoms>', '/bids_root_dir/sub-01/ses-01/func', 'sub-01', 'sub-01', 'ses-01')

    """
    try:
        # Ensure the output directory for functional scans exists.
        os.makedirs(temp_dir, exist_ok=True)
        base_cmd = [
            'dcm2niix',
            '-f', f'{subject_id}_{session_id}_%p', # Naming convention. 
            '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
            '-b', 'y', # Save BIDS metadata to .json sidecar. 
            '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
            '-x', 'y', # Crop images. This will attempt to remove excess neck from 3D acquisitions.
            '-z', 'n', # Do not compress files.
            '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
            '-i', 'n', # Do not ignore derived, localizer and 2D images. 
            '-m', '2', # Merge slices from same series automatically based on modality. 
            '-o', temp_dir,
            input_dir
        ]
        
        # Create a temporary directory for the verbose output run.
        with tempfile.TemporaryDirectory() as temp_dir:# Run the actual conversion without verbose output.
            subprocess.run(base_cmd) #capture_output=False, text=False)
        logging.info(f"dcm2niix conversion completed successfully to {temp_dir}.")

    # Log conversion errors.
    except subprocess.CalledProcessError as e:
        logging.error("dcm2niix conversion failed: %s", e)
        raise
    
    # Log other errors.
    except Exception as e:
        logging.error("An error occurred during dcm2niix conversion: %s", e)
        raise

# Runs the dcm2niix conversion tool to produce verbose output to logfile. 
def run_dcm2niix_verbose(input_dir, temp_dir_verbose, subject_id, session_id, log_file_path):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - temp_dir (str): Directory where the converted NIfTI files will be saved and deleted. 
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    The function logs verbose output to the specified log file. 

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<func_dicoms>, 'temp_dir', 'sub-01', 'ses-01')

    """
    try:
        verbose_cmd = [
        'dcm2niix',
        '-f', f'{subject_id}_{session_id}_%p', # Naming convention. 
        '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
        '-b', 'y', # Save BIDS metadata to .json sidecar. 
        '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
        '-x', 'n', # Do notCrop images. This will attempt to remove excess neck from 3D acquisitions.
        '-z', 'n', # Do not compress files.
        '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
        '-i', 'n', # Do not ignore derived, localizer and 2D images. 
        '-m', '2', # Merge slices from same series automatically based on modality. 
        '-v', 'y', # Print verbose output to logfile.
        '-o', temp_dir_verbose,
        input_dir
    ]
        
        # Create a temporary directory for the verbose output run.
        with tempfile.TemporaryDirectory() as temp_dir_verbose:
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

def main(dicom_root_dir, bids_root_dir):
    """
    Processes MRI data by converting DICOM files to NIfTI format and organizing them according to the BIDS standard.

    This function extracts subject and session IDs from the DICOM directory path, checks for existing NIfTI files, and
    handles the conversion of DICOM to NIfTI format using dcm2niix. It also manages BIDS-compliant metadata and error logging.

    Parameters:
    - dicom_root_dir (str): The root directory containing the DICOM files.
    - bids_root_dir (str): The root directory where the BIDS-compliant NIfTI files and metadata will be stored.
    - args (object): An object containing additional arguments and configurations.

    Workflow:
    1. Extract subject and session IDs from the DICOM directory path.
    2. Check for existing NIfTI files to avoid redundant processing.
    3. Convert DICOM files to NIfTI format using dcm2niix.
    4. Organize converted files in a BIDS-compliant structure.
    5. Add and remove specific metadata using cubids, if installed.
    6. Handle errors and log processing steps.

    Returns:
    None. This function primarily performs file operations and logs its progress.

    Raises:
    - FileNotFoundError: If a required file is not found during processing.
    - Exception: For any other errors encountered during the process.

    Note: This function requires dcm2niix and optionally cubids to be installed and accessible in the system's PATH.
    """
   
    # Extract subject and session IDs from the DICOM directory path.
    subject_id, session_id = extract_subject_session(dicom_root_dir)

    # Specify the exact directory where the NIfTI files will be saved.
    output_dir_func = os.path.join(bids_root_dir, f'{subject_id}', f'{session_id}', 'func')
    os.makedirs(output_dir_func, exist_ok=True)

    # Check if Task fMRI files already exist.
    if check_existing_nifti(output_dir_func, subject_id, session_id):
        print(f"Task fmri NIfTI files already exist: {output_dir_func}")
        return # Skip processing if Task fMRI files already exist.

    try:
        # Specify the exact directory where the DICOM files are located within the root directory
        base_dicom_dir = os.path.join(dicom_root_dir)

        # Setup logging after extracting subject_id and session_id.
        log_file_path = setup_logging(subject_id, session_id, bids_root_dir)
        logging.info(f"Processing TASK FMRI data for subject: {subject_id}, session: {session_id}")

        run_mapping = {
            '': '01',
            '_seqexec_PRE': '00',
            '_seqexec_POST': '07',
            '_1': '01',
            '_2': '02',
            '_3': '03',
            '_4': '04',
            '_5': '05',
            '_6': '06'
        }

        # Check if dcm2niix is installed and accessible in the system's PATH.
        if check_dcm2niix_installed():
            logging.info("Starting DICOM to NIfTI conversion process.")
            for suffix, run in run_mapping.items():
                dicom_dir = os.path.join(base_dicom_dir, f'sms3_TASK{suffix}')
                logging.info(f"Preparing to convert DICOM files in {dicom_dir}")

                with tempfile.TemporaryDirectory() as temp_dir:
                    logging.info(f"Running dcm2niix on {dicom_dir}")
                    run_dcm2niix_output = run_dcm2niix(dicom_dir, temp_dir, subject_id, session_id)
                    logging.info(f"dcm2niix output: {run_dcm2niix_output}")

                    converted_files = os.listdir(temp_dir)
                    logging.info(f"Files after conversion: {converted_files}")
                    if not converted_files:
                        logging.warning(f"No files were converted in {dicom_dir}")
                        continue

                    for old_file in converted_files:
                        try:
                            old_filepath = os.path.join(temp_dir, old_file)
                            new_file = f"{subject_id}_{session_id}_task-learn_run-{run}_bold{os.path.splitext(old_file)[-1]}"
                            new_filepath = os.path.join(output_dir_func, new_file)

                            shutil.move(old_filepath, new_filepath)
                            logging.info(f"Moved {old_file} to {new_filepath}")

                            if new_filepath.endswith('.json') and os.path.isfile(new_filepath):
                                logging.info(f"Updating JSON file with BIDS metadata: {new_filepath}")
                                update_json_file(new_filepath)
                                logging.info("JSON file update successful.")
                        
                        except FileNotFoundError:
                            logging.error(f"File not found during moving process: {old_file}")
                        except Exception as e:
                            logging.error(f"Error processing file {old_file}: {e}")

            if check_cubids_installed():
                logging.info(f"Adding and removing NIfTI metadata for subject: {subject_id}, session: {session_id}")
                run_cubids_add_nifti_info(bids_root_dir)
                run_cubids_remove_metadata_fields(bids_root_dir, ['PatientBirthDate'])
                # Run dcm2niix for verbose output.
                with tempfile.TemporaryDirectory() as temp_dir_verbose:
                    run_dcm2niix_verbose(dicom_dir, temp_dir_verbose, subject_id, session_id, log_file_path)
    
            else:
                logging.error("cubids is not installed. Skipping cubids commands.")
               # Catch error if dcm2niix is not installed.
        else:
            logging.error("dcm2niix is not installed. Cannot proceed with DICOM to NIfTI conversion.")
            return  # Exit the function if dcm2niix is not installed.
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        raise

# Entry point of the script when executed from the command line.
if __name__ == "__main__":
    """
    Script's entry point when executed from the command line. 
    This script processes task fMRI DICOM files, converting them into NIfTI format following 
    the Brain Imaging Data Structure (BIDS) conventions. It handles multiple runs of fMRI data 
    and organizes the output into a BIDS-compliant dataset structure.

    The script requires three command-line arguments:
    - The root directory containing the DICOM directories.
    - The root directory of the BIDS dataset where the converted NIfTI files will be stored.
    - The number of runs (sessions) of fMRI data to process.

    Usage:
        python process_task_learn_to_BIDS.py <dicom_root_dir> <bids_root_dir>
        Example: python process_task_learn_to_BIDS.py /path/to/dicom /path/to/bids 

    The script ensures that the necessary tools (dcm2niix and cubids) are installed and accessible. 
    It performs detailed logging of each step and robust error handling for reliable processing.
    """ 
    
    # Set up an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description='Process DICOM files for TASK FMRI and convert to NIfTI.')

    # Add arguments to the parser.

    # The first argument is the root directory containing the DICOM directories.
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    
    # The second argument is the root directory of the BIDS dataset.
    parser.add_argument('bids_root_dir', type=str, help='Root directory of the BIDS dataset.')

    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Call the main function with the parsed arguments.
    main(args.dicom_root_dir, args.bids_root_dir)


