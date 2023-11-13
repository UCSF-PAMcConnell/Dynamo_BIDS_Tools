"""
process_fmap_epi_to_BIDS.py

Description:
This script processes Echo Planar Imaging (EPI) Reverse Phase Echo (RPE) Field Map DICOM files into NIfTI format following the Brain Imaging Data Structure (BIDS) conventions. 
It automates the conversion using dcm2niix, applies BIDS-compliant metadata processing with cubids, and handles file renaming for fMRI data analysis. 
This technique is essential for correcting distortions in EPI images by acquiring images with opposite phase-encoding polarities (e.g., anterior-posterior and posterior-anterior).

Purpose: 
The primary purpose of this script is to facilitate the correction of EPI-induced distortions in fMRI images. 
It aids in comparing phase-encoded images to estimate and correct voxel displacement.

Process Overview:
1. Extracts subject and session IDs from DICOM file paths.
2. Converts DICOM files to NIfTI format using dcm2niix.
3. Adds and updates JSON metadata for BIDS compliance.
4. Optionally removes specific files and unwanted metadata fields.

Usage:
Run the script from the command line with the following format:
python process_fmap_epi_to_BIDS.py <dicom_root_dir> <bids_root_dir>

Example:
python process_fmap_epi_to_BIDS.py /path/to/dicom_root_dir /path/to/bids_root_dir

Author: PAMcConnell
Created on: 20231112
Last Modified: 20231112
Version: 1.0.0

License: MIT License

Dependencies:
- Python 3.12
- pydicom for reading DICOM files.
- dcm2niix (command-line tool) for DICOM to NIfTI conversion.
- CuBIDS (command-line tool) for BIDS metadata processing.
- Standard Python libraries: tempfile, os, logging, subprocess, argparse, re, sys, json, glob.

Environment Setup:
- Ensure Python 3.12, dcm2niix, and cubids are installed.
- Use conda for installation: `conda install -c conda-forge dcm2niix pydicom`.
- Install cubids with `pip install cubids` or follow the official documentation.
- An environment.yml file (e.g., datalad.yml) is provided for setting up with Conda.

Error Handling and Logging:
- The script includes detailed error handling and logging mechanisms to provide insights into the processing steps and assist in troubleshooting.

Change Log:
- 20231112: Initial version. Implemented DICOM to NIfTI conversion and BIDS metadata processing.
"""

import os                     # Used for operating system dependent functionalities like file path manipulation.
import logging                # Logging library, for tracking events that happen when running the software.
import tempfile               # Generate temporary files and directories.
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

# Checks if AP EPI FIELD MAP NIfTI files already exist in the specified BIDS output directory.
def check_existing_nifti_AP(output_dir_AP, subject_id, session_id):
    """
    Parameters:
    - output_dir_AP (str): The BIDS output directory where NIfTI files are stored.
    - subject_id (str): The subject ID.
    - session_id (str): The session ID.

    Returns:
    - bool: True if AP EPI FIELD MAP NIfTI files exist, False otherwise.
    """
    expected_nifti_file = os.path.join(output_dir_AP, f'{subject_id}_{session_id}_dir-AP_epi.nii')
    if os.path.isfile(expected_nifti_file):
        print(f"AP EPI FIELD MAP NIfTI file already exists: {expected_nifti_file}")
        return True
    else:
        return False

# Checks if PA EPI FIELD MAP NIfTI files already exist in the specified BIDS output directory.
def check_existing_nifti_PA(output_dir_PA, subject_id, session_id):
    """
    Parameters:
    - output_dir_PA (str): The BIDS output directory where NIfTI files are stored.
    - subject_id (str): The subject ID.
    - session_id (str): The session ID.

    Returns:
    - bool: True if PA EPI FIELD MAP NIfTI files exist, False otherwise.
    """
    expected_nifti_file = os.path.join(output_dir_PA, f'{subject_id}_{session_id}_dir-PA_epi.nii')
    if os.path.isfile(expected_nifti_file):
        print(f"PA EPI FIELD MAP NIfTI file already exists: {expected_nifti_file}")
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
    """
    Parameters:
    - dicom_dir (str): Path to the directory containing DICOM files.

    Returns:
    - list: A list of DICOM header objects from pydicom.

    This function scans the specified directory for DICOM files (prefixed with 'MR.' with no file extension)
    and reads their headers using pydicom. It returns a list of DICOM header objects, 
    which can be used for further processing and analysis.

    Error handling is implemented to catch issues with reading files or invalid DICOM files.

    Usage Example:
    dicom_headers = read_dicom_headers('/path/to/dicom_dir')

    Dependencies:
    - pydicom module for reading DICOM files.
    - os module for directory and file operations.
    """
    try:
        # Identify DICOM files in the directory.
        dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                       if os.path.isfile(os.path.join(dicom_dir, f)) and f.startswith('MR.')]

        # Read the headers of each DICOM file.
        dicom_headers = [pydicom.dcmread(df, force=True) for df in dicom_files]
        logging.info(f"Read headers of {len(dicom_headers)} DICOM files from {dicom_dir}")

        return dicom_headers
    
    # Catch directory not found error.
    except FileNotFoundError as e:
        logging.error(f"Directory not found: {dicom_dir}. Error: {e}")
        raise
    
    # Catch issues with reading dicom files. 
    except Exception as e:
        logging.error(f"Error reading DICOM files from {dicom_dir}. Error: {e}")
        raise

# Updates the JSON sidecar file with specific fields required for BIDS compliance in AP EPI FIELD MAP datasets.
def update_json_file_AP(json_filepath_AP, session_id,intended_for=None):
    """
    Parameters:
    - json_filepath (str): Path to the JSON sidecar file.

    This function updates the specified JSON file with fields relevant to EPI FIELD MAP imaging.

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
        with open(json_filepath_AP, 'r+') as file:
            data = json.load(file)
            
            # Update with specific AP EPI FIELD MAP metadata
            if session_id == 'ses-1':
                data['B0FieldIdentifier'] = "*epse2d1_104"
                logging.info(f"Updated B0FieldIdentifier to {data['B0FieldIdentifier']}")
            else: 
                data['B0FieldIdentifier'] = "*epfid2d1_96"
                logging.info(f"Updated B0FieldIdentifier to {data['B0FieldIdentifier']}")
            
            data['IntendedFor'] = intended_for
            logging.info(f"Updated IntendedFor to {data['IntendedFor']}")

            # Write back the updated data and truncate the file to the new data length.
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

        logging.info(f"Updated JSON file at {json_filepath_AP} with AP EPI FIELD MAP-specific metadata.")
    
    # Catch issues with reading or writing to the JSON file.
    except IOError as e:
        logging.error(f"Error reading or writing to JSON file at {json_filepath_AP}. Error: {e}")
        raise

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON data in file at {json_filepath_AP}. Error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred while updating JSON file at {json_filepath_AP}. Error: {e}")
        raise

# Updates the JSON sidecar file with specific fields required for BIDS compliance in PA EPI FIELD MAP datasets.
def update_json_file_PA(json_filepath_PA, session_id, intended_for=None):
    """
    Parameters:
    - json_filepath (str): Path to the JSON sidecar file.

    This function updates the specified JSON file with fields relevant to EPI FIELD MAP imaging.

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
        with open(json_filepath_PA, 'r+') as file:
            data = json.load(file)
            
            # Update with specific AP EPI FIELD MAP metadata
            if session_id == 'ses-1':
                data['B0FieldIdentifier'] = "*epse2d1_104"
                logging.info(f"Updated B0FieldIdentifier to {data['B0FieldIdentifier']}")
            else: 
                data['B0FieldIdentifier'] = "*epfid2d1_96"
                logging.info(f"Updated B0FieldIdentifier to {data['B0FieldIdentifier']}")
            
            data['IntendedFor'] = intended_for
            logging.info(f"Updated IntendedFor to {data['IntendedFor']}")

            # Write back the updated data and truncate the file to the new data length.
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

        logging.info(f"Updated JSON file at {json_filepath_PA} with EPI FIELD MAP-specific metadata.")
    
    # Catch issues with reading or writing to the JSON file.
    except IOError as e:
        logging.error(f"Error reading or writing to JSON file at {json_filepath_PA}. Error: {e}")
        raise

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON data in file at {json_filepath_PA}. Error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred while updating JSON file at {json_filepath_PA}. Error: {e}")
        raise

# Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format.
def run_dcm2niix_AP(input_dir_AP, output_dir_AP, subject_id, session_id):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - output_dir_fmap (str): Directory where the converted NIfTI files will be saved.
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    This function uses the dcm2niix tool to convert DICOM files into NIfTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. 
    The function assumes that dcm2niix is installed and accessible in the system's PATH.

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<fmap_dicoms>', '/bids_root_dir/sub-01/ses-01/fmap', 'sub-01', 'ses-01')

    """
    try:
        # Ensure the output directory for anatomical scans exists.
        os.makedirs(output_dir_AP, exist_ok=True)
        base_cmd = [
            'dcm2niix',
            '-f', f'{subject_id}_{session_id}_dir-AP_epi', # Naming convention. 
            '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
            '-b', 'y', # Save BIDS metadata to .json sidecar. 
            '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
            '-x', 'n', # Do notCrop images. This will attempt to remove excess neck from 3D acquisitions.
            '-z', 'n', # Do not compress files.
            '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
            '-i', 'n', # Do not ignore derived, localizer and 2D images. 
            '-m', '2', # Merge slices from same series automatically based on modality. 
            '-o', output_dir_AP,
            input_dir_AP
        ]
    
        # Run the actual conversion without verbose output.
        subprocess.run(base_cmd) #capture_output=False, text=False)
        logging.info(f"dcm2niix conversion completed successfully to {output_dir_AP}.")

    # Log conversion errors.
    except subprocess.CalledProcessError as e:
        logging.error("dcm2niix conversion failed: %s", e)
        raise
    
    # Log other errors.
    except Exception as e:
        logging.error("An error occurred during dcm2niix conversion: %s", e)
        raise

# Runs the dcm2niix conversion tool to produce verbose output to logfile. 
def run_dcm2niix_verbose_AP(input_dir_AP, temp_dir, subject_id, session_id, log_file_path):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - temp_dir (str): Directory where the converted NIfTI files will be saved and deleted. 
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    The function logs verbose output to the specified log file. 

    Usage Example:
    run_dcm2niix('/path/to/dicom', 'temp_dir', 'sub-01', 'ses-01')

    """
    try:
        verbose_cmd = [
        'dcm2niix',
        '-f', f'{subject_id}_{session_id}_dir-AP_epi', # Naming convention. 
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
        input_dir_AP
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

# Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format.
def run_dcm2niix_PA(input_dir_PA, output_dir_PA, subject_id, session_id):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - output_dir_fmap (str): Directory where the converted NIfTI files will be saved.
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    This function uses the dcm2niix tool to convert DICOM files into NIfTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. 
    The function assumes that dcm2niix is installed and accessible in the system's PATH.

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<fmap_dicoms>', '/bids_root_dir/sub-01/ses-01/fmap', 'sub-01', 'ses-01')

    """
    try:
        # Ensure the output directory for anatomical scans exists.
        os.makedirs(output_dir_PA, exist_ok=True)
        base_cmd = [
            'dcm2niix',
            '-f', f'{subject_id}_{session_id}_dir-PA_epi', # Naming convention. 
            '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
            '-b', 'y', # Save BIDS metadata to .json sidecar. 
            '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
            '-x', 'n', # Do notCrop images. This will attempt to remove excess neck from 3D acquisitions.
            '-z', 'n', # Do not compress files.
            '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
            '-i', 'n', # Do not ignore derived, localizer and 2D images. 
            '-m', '2', # Merge slices from same series automatically based on modality. 
            '-o', output_dir_PA,
            input_dir_PA
        ]
    
        # Run the actual conversion without verbose output.
        subprocess.run(base_cmd) #capture_output=False, text=False)
        logging.info(f"dcm2niix conversion completed successfully to {output_dir_PA}.")

    # Log conversion errors.
    except subprocess.CalledProcessError as e:
        logging.error("dcm2niix conversion failed: %s", e)
        raise
    
    # Log other errors.
    except Exception as e:
        logging.error("An error occurred during dcm2niix conversion: %s", e)
        raise

# Runs the dcm2niix conversion tool to produce verbose output to logfile. 
def run_dcm2niix_verbose_PA(input_dir_PA, temp_dir, subject_id, session_id, log_file_path):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - temp_dir (str): Directory where the converted NIfTI files will be saved and deleted. 
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    The function logs verbose output to the specified log file. 

    Usage Example:
    run_dcm2niix('/path/to/dicom', 'temp_dir', 'sub-01', 'ses-01')

    """
    try:
        verbose_cmd = [
        'dcm2niix',
        '-f', f'{subject_id}_{session_id}_dir-PA_epi', # Naming convention. 
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
        input_dir_PA
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

# Main function to orchestrate the process of converting EPI Field Map DICOM files to BIDS-compliant NIfTI format.
def main(dicom_root_dir, bids_root_dir):
    """
    Orchestrate the process of converting EPI Field Map DICOM files to BIDS-compliant NIfTI format.

    This function handles the conversion of Anterior-Posterior (AP) and Posterior-Anterior (PA) 
    EPI Field Map DICOM files. It ensures the converted NIfTI files are appropriately named and 
    stored in the BIDS dataset, following the BIDS naming conventions and directory structure.

    Steps:
    1. Extracts subject and session IDs from DICOM directory paths.
    2. Checks for the existence of converted NIfTI files to prevent redundant processing.
    3. Converts AP and PA DICOM files to NIfTI format using dcm2niix.
    4. Updates JSON metadata for the NIfTI files to adhere to BIDS specifications.
    5. Removes unwanted metadata fields and specific files based on naming patterns.
    
    Parameters:
    - dicom_root_dir (str): Path to the directory containing the DICOM files.
    - bids_root_dir (str): Path to the BIDS dataset root directory.

    The function requires the dcm2niix tool for DICOM to NIfTI conversion and cubids for BIDS 
    metadata processing. It also handles specific file removals post-conversion.

    Usage Example:
    main('/path/to/dicom_root_dir', '/path/to/bids_root_dir')

    Dependencies:
    - External tools: dcm2niix, cubids.
    - Python libraries: os, logging, tempfile.
    """

    # Extract subject and session IDs from the DICOM directory path.
    subject_id, session_id = extract_subject_session(dicom_root_dir)

    # Specify the exact directory where the NIfTI files will be saved.
    output_dir_AP = os.path.join(bids_root_dir, f'{subject_id}', f'{session_id}', 'fmap')
    output_dir_PA = os.path.join(bids_root_dir, f'{subject_id}', f'{session_id}', 'fmap')

    # Check if AP EPI Field Map NIfTI files already exist.
    if check_existing_nifti_AP(output_dir_AP, subject_id, session_id):
        print(f"AP EPI Field Map NIfTI files already exist for subject: {subject_id}, session: {session_id}")
        return # Exit the function if NIfTI files already exist.

    # Check if PA EPI Field Map NIfTI files already exist.
    if check_existing_nifti_AP(output_dir_PA, subject_id, session_id):
        print(f"PA EPI Field Map NIfTI files already exist for subject: {subject_id}, session: {session_id}")
        return # Exit the function if NIfTI files already exist.

    # Otherwise:
    try:
        # Setup logging after extracting subject_id and session_id.
        log_file_path = setup_logging(subject_id, session_id, bids_root_dir)
        logging.info(f"Processing GRE Field Map data for subject: {subject_id}, session: {session_id}")

        # Specify the exact directory where the DICOM files are located.
        if session_id == 'ses-1':
            dicom_dir_AP = os.path.join(dicom_root_dir, 'SpinEchoFieldMap_AP')
            dicom_dir_PA = os.path.join(dicom_root_dir, 'SpinEchoFieldMap_PA')
        else:
            dicom_dir_AP = os.path.join(dicom_root_dir, 'sms3_distortionmap_AP')
            dicom_dir_PA = os.path.join(dicom_root_dir, 'sms3_distortionmap_PA')

        # Check if dcm2niix is installed and accessible in the system's PATH.
        if check_dcm2niix_installed():
            
            # Run dcm2niix for AP DICOM to NIfTI conversion.
            run_dcm2niix_AP(dicom_dir_AP, output_dir_AP, subject_id, session_id)

            # Check if cubids is installed
            if check_cubids_installed():
                
                # Update JSON file with necessary BIDS metadata
                # Intended to correct magnetic field distortion in functional MRI images.
                if session_id == 'ses-1':
                    intended_for = [
                    f"{session_id}/func/{subject_id}_{session_id}_task-rest_run-01_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-rest_run-02_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-rest_run-03_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-rest_run-04_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-rest_run-01_sbref.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-rest_run-02_sbref.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-rest_run-03_sbref.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-rest_run-04_sbref.nii",
                    f"{session_id}/perf/{subject_id}_{session_id}_asl.nii"
                    ]
                else:
                    intended_for = [
                    f"{session_id}/func/{subject_id}_{session_id}_task-learn_run-00_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-learn_run-01_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-learn_run-02_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-learn_run-03_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-learn_run-04_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-learn_run-05_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-learn_run-06_bold.nii",
                    f"{session_id}/func/{subject_id}_{session_id}_task-learn_run-07_bold.nii",
                    ]

                # Run cubids commands to add NIfTI metadata.
                logging.info(f"Adding BIDS metadata for subject: {subject_id}, session: {session_id}")
                run_cubids_add_nifti_info(bids_root_dir)

                # Construct the JSON file name
                json_filepath_AP = os.path.join(output_dir_AP, f'{subject_id}_{session_id}_dir-AP_epi.json')

                # Call the function to update the JSON file
                update_json_file_AP(json_filepath_AP, session_id, intended_for)

                # Run cubids commands to remove metadata fields.
                logging.info(f"Removing metadata fields for subject: {subject_id}, session: {session_id}")
                run_cubids_remove_metadata_fields(bids_root_dir, ['PatientBirthDate'])
            
                # Run dcm2niix for PA DICOM to NIfTI conversion.    
                run_dcm2niix_PA(dicom_dir_PA, output_dir_PA, subject_id, session_id)
            
                # Run cubids commands to add NIfTI metadata.
                logging.info(f"Adding BIDS metadata for subject: {subject_id}, session: {session_id}")
                run_cubids_add_nifti_info(bids_root_dir)

                # Construct the JSON file name
                json_filepath_PA = os.path.join(output_dir_PA, f'{subject_id}_{session_id}_dir-PA_epi.json')

                # Call the function to update the JSON file
                update_json_file_PA(json_filepath_PA, session_id, intended_for)
                
                # Run cubids commands to remove BIDS metadata from top up files.
                logging.info(f"Removing BIDS metadata from {subject_id}_{session_id}_acq-topup_dir-PA_epi.nii")
                run_cubids_remove_metadata_fields(bids_root_dir, ['PatientBirthDate'])

                # Run dcm2niix for verbose output.
                with tempfile.TemporaryDirectory() as temp_dir:
                    run_dcm2niix_verbose_AP(dicom_dir_AP, temp_dir, subject_id, session_id, log_file_path)
                    run_dcm2niix_verbose_PA(dicom_dir_PA, temp_dir, subject_id, session_id, log_file_path)
                
                # Loop through all files in the specified directory to remove EPI_phase images.
                for filename in os.listdir(output_dir_PA):
                    # Check if 'epi_ph' is in the filename
                    if 'epi_ph' in filename:
                        # Construct full file path
                        file_path = os.path.join(output_dir_PA, filename)
                        # Remove the file
                        os.remove(file_path)
                        print(f"Removed file: {file_path}")

            # Catch error if cubids is not installed.
            else:
                logging.error("cubids is not installed. Skipping cubids commands.")
                return  # Exit the function if cubids is not installed.
        
        # Catch error if dcm2niix is not installed.
        else:
            logging.error("dcm2niix is not installed. Cannot proceed with DICOM to NIfTI conversion.")
            return  # Exit the function if dcm2niix is not installed.
        
    # Log other errors. 
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        raise

# Main code execution starts here when the script is run
if __name__ == "__main__":
    """
    Entry point of the script when executed from the command line.

    Parses command-line arguments to determine the directories for DICOM files and BIDS dataset.

    Usage:
    process_fmap_EPI_to_BIDS.py <dicom_root_dir> <bids_root_dir>
   
   """  
    
    # Set up an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description='Process DICOM files and convert to NIfTI.')
    
     # Add arguments to the parser.

    # The first argument is the root directory containing the DICOM directories.
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    
    # The second argument is the root directory of the BIDS dataset.
    parser.add_argument('bids_root_dir', type=str, help='Root directory of the BIDS dataset.')
    
    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Call the main function with the parsed arguments.
    main(args.dicom_root_dir, args.bids_root_dir)
    
    