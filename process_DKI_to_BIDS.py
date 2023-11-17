"""
process_DKI_to_BIDS.py

Description:
This script processes Diffusion Kurtosis Imaging (DKI) DICOM files into NIFTI format following the Brain Imaging Data Structure (BIDS) conventions. 
It includes functionalities for DICOM to NIFTI conversion using dcm2niix and additional BIDS-compliant metadata processing with cubids.
The script checks for the installation of dcm2niix, pydeface, and cubids 
before executing relevant commands. It also handles file renaming.

Usage:
python process_DKI_to_BIDS.py <dicom_root_dir> <bids_root_dir>
e.g.,
python process_DKI_to_BIDS.py /path/to/dicom_root_dir /path/to/bids_root_dir 

Author: PAMcConnell
Created on: 20231112
Last Modified: 20231112

License: MIT License

Dependencies:
- Python 3.12
- pydicom for reading DICOM files.
- dcm2niix (command-line tool) https://github.com/rordenlab/dcm2niix
- CuBIDS (command-line tool) https://cubids.readthedocs.io/en/latest/index.html
- Standard Python libraries: tempfile, os, logging, subprocess, argparse, re, sys, json, glob.

Environment Setup:
- Ensure Python 3.12, dcm2niix and cubids are installed in your environment.
- You can install dcm2niix using conda: `conda install -c conda-forge dcm2niix pydicom`.
- Install cubids following the instructions provided in its documentation.
- Try 'pip install cubids'
- To set up the required environment, you can use the provided environment.yml file with Conda. <datalad.yml>

Change Log:
- 20231112: Initial version

*** NOTE: Verify correct "IntendedFor" Path ***

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

# Extracts the B0FieldIdentifier from DICOM tags in the first file of a directory.
def extract_b0_field_identifier(dicom_dir_topup):
    """
    Extracts the B0FieldIdentifier (assumed as 'SequenceName') from DICOM tags 
    in the first DICOM file found in a specified directory.

    This function iterates over files in the given directory and reads the first 
    DICOM file it finds to extract the specified tag.

    Parameters:
    dicom_dir_topup (str): Path to the directory containing DICOM files.
    
    Returns:
    str: The value of 'SequenceName' from the first DICOM file in the directory,
         or a message indicating that no DICOM files were found.

    Usage example:
    b0_field_identifier = extract_b0_field_identifier('/path/to/dicom/dir')
    print(b0_field_identifier)

    """
    try:
        # Iterate over files in the directory
        for filename in os.listdir(dicom_dir_topup):
            file_path = os.path.join(dicom_dir_topup, filename)
        
        # Check if the file is a DICOM file
        if pydicom.misc.is_dicom(file_path):
            dicom_data = pydicom.dcmread(file_path)
            
            # Extract 'SequenceName' from the DICOM file
            sequence_name = dicom_data.get("SequenceName", "Tag not found")
            logging.info(f"Extracted SequenceName: {sequence_name}")
            return sequence_name
        
        # Log if no DICOM files are found in the directory
        logging.warning("No DICOM files found in the directory: " + dicom_dir_topup)
        return "No DICOM files found in the directory."
    
    # Catch any errors and log them.
    except Exception as e:
        logging.error("Error occurred while extracting B0FieldIdentifier: " + str(e))
        raise

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

# Checks if DKI NIFTI files already exist in the specified BIDS output directory.
def check_existing_nifti_dwi(output_dir_dwi, subject_id, session_id):
    """
    Checks if Diffusion Kurtosis Imaging (DKI) NIFTI files already exist in the 
    specified BIDS (Brain Imaging Data Structure) output directory. This function 
    is designed to prevent redundant data processing by verifying the presence of 
    expected NIFTI files.

    Parameters:
    - output_dir_dwi (str): The BIDS output directory where NIFTI files are stored.
    - subject_id (str): The subject ID, used in the naming convention of the files.
    - session_id (str): The session ID, used in the naming convention of the files.

    Returns:
    - bool: True if the DKI NIFTI file exists, False otherwise.

    Usage example:
    file_exists = check_existing_nifti_dwi('/path/to/output/dir', 'sub-01', 'ses-01')
    if file_exists:
        print("NIFTI file already exists.")
    """
    try:
        # Construct the expected file path based on BIDS naming convention
        expected_nifti_file_dwi = os.path.join(output_dir_dwi, f'{subject_id}_{session_id}_dir-AP_dwi.nii')
        
        # Check if the expected NIFTI file exists
        if os.path.isfile(expected_nifti_file_dwi):
            print(f"DKI NIFTI file already exists, skipping...: {expected_nifti_file_dwi}")
            return True
        else:
            logging.info(f"No DKI NIFTI file found, proceeding with processing: {expected_nifti_file_dwi}")
            return False
        
    # Catch and log any exceptions.
    except Exception as e:
        logging.error(f"Error occurred while checking for existing DKI NIFTI file: {str(e)}")
        raise

# Checks if Top Up NIFTI files already exist in the specified BIDS output directory.
def check_existing_nifti_topup(output_dir_topup, subject_id, session_id):
    """
    Checks if Top Up NIFTI files already exist in the specified BIDS (Brain Imaging Data Structure) 
    output directory. The function is designed to avoid redundant data processing by ensuring that 
    the necessary NIFTI files for a given subject and session have not been previously generated.

    Parameters:
    - output_dir_topup (str): The BIDS output directory where NIFTI files are stored.
    - subject_id (str): The subject ID, used in the naming convention of the files.
    - session_id (str): The session ID, used in the naming convention of the files.

    Returns:
    - bool: True if the Top Up NIFTI file exists, False otherwise.

    Usage example:
    file_exists = check_existing_nifti_topup('/path/to/output/dir', 'sub-01', 'ses-01')
    if file_exists:
        print("Top Up NIFTI file already exists.")
    """

    try:
        # Construct the expected file path based on BIDS naming convention
        expected_nifti_file_topup= os.path.join(output_dir_topup, f'{subject_id}_{session_id}_acq-topup_dir-PA_epi.nii')
        
        # Check if the expected NIFTI file exists
        if os.path.isfile(expected_nifti_file_topup):
            print(f"Top Up NIFTI file already exists, skipping...: {expected_nifti_file_topup}")
            return True
        else:
            print(f"No Top Up NIFTI file found, proceeding with processing: {expected_nifti_file_topup}")
            return False
        
    # Catch and log any exceptions.
    except Exception as e:
        logging.error(f"Error occurred while checking for existing Top Up NIFTI file: {str(e)}")
        raise
    
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

# Updates the DKI JSON sidecar file with specific fields required for BIDS compliance in DKI datasets.
def update_json_file_dwi(json_filepath_dwi, subject_id, session_id, dicom_dir_topup):
    """
    Updates the specified JSON sidecar file with fields relevant to Diffusion Kurtosis Imaging (DKI).
    This function reads the existing JSON file, adds or updates specific metadata fields, and then 
    writes the changes back to the file. The updates ensure compliance with the BIDS standard for DKI/DWI data.

    Parameters:
    - json_filepath (str): Path to the JSON sidecar file.
    - dicom_dir_topup (str): Directory containing DICOM files used for extracting B0 field information.

    Usage Example:
    update_json_file('/path/to/sidecar.json', '/path/to/dicom/dir')

    Dependencies:
    - json module for reading and writing JSON files.
    - Logging for logging information and errors.
    - Function 'extract_b0_field_identifier' for extracting B0 field information from DICOM files.

    Note: Refer to MR protocol documentation in /doc/MR_protocols for more information about metadata fields.
    """

    try:
        with open(json_filepath_dwi, 'r+') as file:
            data = json.load(file)
            
            # Updated DKI specific metadata for topup field map correction.
            data['B0FieldSource'] = extract_b0_field_identifier(dicom_dir_topup)

            # Write back the updated data and truncate the file to the new data length.
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

        logging.info(f"Updated JSON file at {json_filepath_dwi} with DKI-specific metadata.")
    
    # Catch issues with reading or writing to the JSON file.
    except IOError as e:
        logging.error(f"Error reading or writing to JSON file at {json_filepath_dwi}. Error: {e}")
        raise

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON data in file at {json_filepath_dwi}. Error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred while updating JSON file at {json_filepath_dwi}. Error: {e}")
        raise

# Updates the DKI JSON sidecar file with specific fields required for BIDS compliance in DKI datasets.
def update_json_file_topup(json_filepath_topup, subject_id, session_id, dicom_dir_topup):
    """
    Updates the specified JSON sidecar file with fields relevant to Diffusion Kurtosis Imaging (DKI)
    and field map correction (Top Up). The function adds or updates the 'B0FieldIdentifier' and 'IntendedFor'
    fields in the JSON file, ensuring compliance with the BIDS standard for DKI/DWI data.

    Parameters:
    - json_filepath_topup (str): Path to the JSON sidecar file for Top Up correction.
    - dicom_dir_topup (str): Directory containing DICOM files used for extracting B0 field information.
    - subject_id (str): Subject ID, used to construct the 'IntendedFor' field.
    - session_id (str): Session ID, used to construct the 'IntendedFor' field.

    Usage Example:
    update_json_file('/path/to/topup/sidecar.json', '/path/to/dicom/dir/topup', 'sub-01', 'ses-01')

    Dependencies:
    - json module for reading and writing JSON files.
    - Logging for logging information and errors.
    - Function 'extract_b0_field_identifier' for extracting B0 field information from DICOM files.

    Note: Refer to MR protocol documentation in /doc/MR_protocols for additional metadata field information.
    """
    try:
        with open(json_filepath_topup, 'r+') as file:
            data = json.load(file)
            
            # Updated DKI specific metadata for topup field map correction.
            data['B0FieldIdentifier'] = extract_b0_field_identifier(dicom_dir_topup)
            data['IntendedFor'] = (f"{session_id}/dwi/{subject_id}_{session_id}_dir-AP_dwi.nii")

            # Write back the updated data and truncate the file to the new data length.
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

        logging.info(f"Updated JSON file at {json_filepath_topup} with DKI-specific metadata.")
    
    # Catch issues with reading or writing to the JSON file.
    except IOError as e:
        logging.error(f"Error reading or writing to JSON file at {json_filepath_topup}. Error: {e}")
        raise

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON data in file at {json_filepath_topup}. Error: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred while updating JSON file at {json_filepath_topup}. Error: {e}")
        raise

# Runs the dcm2niix conversion tool to convert DICOM files to NIFTI format.
def run_dcm2niix_dwi(input_dir, output_dir_dwi, subject_id, session_id):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - output_dir_dwi (str): Directory where the converted NIFTI files will be saved.
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    This function uses the dcm2niix tool to convert DICOM files into NIFTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. 
    The function assumes that dcm2niix is installed and accessible in the system's PATH.

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<dwi_dicoms>', '/bids_root_dir/sub-01/ses-01/dwi', 'sub-01', 'ses-01')

    """
    try:
        # Ensure the output directory for anatomical scans exists.
        os.makedirs(output_dir_dwi, exist_ok=True)
        base_cmd = [
            'dcm2niix',
            '-f', f'{subject_id}_{session_id}_dir-AP_dwi', # Naming convention. 
            '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
            '-b', 'y', # Save BIDS metadata to .json sidecar. 
            '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
            '-x', 'n', # Do notCrop images. This will attempt to remove excess neck from 3D acquisitions.
            '-z', 'n', # Do not compress files.
            '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
            '-i', 'n', # Do not ignore derived, localizer and 2D images. 
            '-m', '2', # Merge slices from same series automatically based on modality. 
            '-o', output_dir_dwi,
            input_dir
        ]

        # Run the actual conversion without verbose output.
        subprocess.run(base_cmd) #capture_output=False, text=False)
        logging.info(f"dcm2niix conversion completed successfully to {output_dir_dwi}.")

    # Log conversion errors.
    except subprocess.CalledProcessError as e:
        logging.error("dcm2niix conversion failed: %s", e)
        raise
    
    # Log other errors.
    except Exception as e:
        logging.error("An error occurred during dcm2niix conversion: %s", e)
        raise

# Runs the dcm2niix conversion tool to produce verbose output to logfile. 
def run_dcm2niix_verbose_dwi(input_dir, temp_dir, subject_id, session_id, log_file_path):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - temp_dir (str): Directory where the converted NIFTI files will be saved and deleted. 
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    The function logs verbose output to the specified log file. 

    Usage Example:
    run_dcm2niix('/path/to/dicom', 'temp_dir', 'sub-01', 'ses-01')

    """
    try:
        verbose_cmd = [
        'dcm2niix',
        '-f', f'{subject_id}_{session_id}_dir-AP_dwi', # Naming convention. 
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

# Runs the dcm2niix conversion tool to convert DICOM files to NIFTI format.
def run_dcm2niix_topup(input_dir, output_dir_topup, subject_id, session_id):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - output_dir_dwi (str): Directory where the converted NIFTI files will be saved.
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    This function uses the dcm2niix tool to convert DICOM files into NIFTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. 
    The function assumes that dcm2niix is installed and accessible in the system's PATH.

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<dwi_dicoms>', '/bids_root_dir/sub-01/ses-01/dwi', 'sub-01', 'ses-01')

    """
    try:
        # Ensure the output directory for anatomical scans exists.
        os.makedirs(output_dir_topup, exist_ok=True)
        base_cmd = [
            'dcm2niix',
            '-f', f'{subject_id}_{session_id}_acq-topup_dir-PA_epi', # Naming convention. 
            '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
            '-b', 'y', # Save BIDS metadata to .json sidecar. 
            '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
            '-x', 'n', # Do notCrop images. This will attempt to remove excess neck from 3D acquisitions.
            '-z', 'n', # Do not compress files.
            '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
            '-i', 'n', # Do not ignore derived, localizer and 2D images. 
            '-m', '2', # Merge slices from same series automatically based on modality. 
            '-o', output_dir_topup,
            input_dir
        ]

        # Run the actual conversion without verbose output.
        subprocess.run(base_cmd) #capture_output=False, text=False)
        logging.info(f"dcm2niix conversion completed successfully to {output_dir_topup}.")

    # Log conversion errors.
    except subprocess.CalledProcessError as e:
        logging.error("dcm2niix conversion failed: %s", e)
        raise
    
    # Log other errors.
    except Exception as e:
        logging.error("An error occurred during dcm2niix conversion: %s", e)
        raise

# Runs the dcm2niix conversion tool to produce verbose output to logfile. 
def run_dcm2niix_verbose_topup(input_dir, temp_dir, subject_id, session_id, log_file_path):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - temp_dir (str): Directory where the converted NIFTI files will be saved and deleted. 
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    The function logs verbose output to the specified log file. 

    Usage Example:
    run_dcm2niix('/path/to/dicom', 'temp_dir', 'sub-01', 'ses-01')

    """
    try:
        verbose_cmd = [
        'dcm2niix',
        '-f', f'{subject_id}_{session_id}_acq-topup_dir-PA_epi', # Naming convention. 
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
def main(dicom_root_dir,bids_root_dir):
    """
    Main function for orchestrating the conversion of DKI DICOM files to NIFTI format following BIDS conventions.
    This function manages the entire conversion workflow, from checking pre-existing files to executing conversion
    and metadata handling tasks.

    Parameters:
    - dicom_root_dir (str): Path to the root directory containing the DKI DICOM files.
    - bids_root_dir (str): Path to the BIDS dataset root directory.

    The function performs the following steps:
    1. Extracts subject and session IDs from the DICOM directory path.
    2. Sets up logging for detailed record-keeping of the process.
    3. Checks if NIFTI files already exist to avoid redundant processing.
    4. Runs dcm2niix for DICOM to NIFTI conversion if necessary.
    5. Executes cubids commands for BIDS-compliant metadata processing.

    Usage Example:
    main('/path/to/dicom_root_dir', '/path/to/bids_root_dir')
    """

    # Extract subject and session IDs from the DICOM directory path.
    subject_id, session_id = extract_subject_session(dicom_root_dir)
    
    # Specify the exact directory where the NIFTI files will be saved.
    output_dir_dwi = os.path.join(bids_root_dir, f'{subject_id}', f'{session_id}', 'dwi')
    output_dir_topup = os.path.join(bids_root_dir, f'{subject_id}', f'{session_id}', 'fmap')

    # Check if DKI NIFTI files already exist.
    if check_existing_nifti_dwi(output_dir_dwi, subject_id, session_id):
        return # Exit the function if DWI NIFTI files already exist.
    
    # Check if TOP-UP NIFTI files already exist.
    if check_existing_nifti_topup(output_dir_topup, subject_id, session_id):
        return   
    
    # Otherwise:   
    try:
        # Setup logging after extracting subject_id and session_id.
        log_file_path = setup_logging(subject_id, session_id, bids_root_dir)
        logging.info(f"Processing DKI data for subject: {subject_id}, session: {session_id}")

        # Specify the exact directory where the DICOM files are located
        dicom_dir_dwi = os.path.join(dicom_root_dir, 'DKI_BIPOLAR_2.5mm_64dir_58slices')
        dicom_dir_topup = os.path.join(dicom_root_dir, 'DKI_BIPOLAR_2.5mm_64dir_58slices_TOP_UP_PA')

        # Check if dcm2niix is installed and accessible in the system's PATH.
        if check_dcm2niix_installed():
                
                # Run dcm2niix for DKI DICOM to NIFTI conversion.
                run_dcm2niix_dwi(dicom_dir_dwi, output_dir_dwi, subject_id, session_id)
                
                # Check if cubids is installed.
                if check_cubids_installed():
                
                    # Run cubids commands to add necessary BIDS metadata to DKI files.
                    logging.info(f"Adding BIDS metadata to {subject_id}_{session_id}_dir-AP_dwi.nii")
                    run_cubids_add_nifti_info(bids_root_dir)
                
                    # Update JSON files with necessary BIDS metadata
                    json_filepath_dwi = os.path.join(output_dir_dwi, f'{subject_id}_{session_id}_dir-AP_dwi.json')
                    update_json_file_dwi(json_filepath_dwi, subject_id, session_id, dicom_dir_topup)
                    
                    # Run cubids commands to remove BIDS metadata from DKI files.
                    logging.info(f"Removing BIDS metadata from {subject_id}_{session_id}_dir-AP_dwi.nii")
                    run_cubids_remove_metadata_fields(bids_root_dir, ['PatientBirthDate'])
                
                    # Run dcm2niix for top up DICOM to NIFTI conversion.    
                    run_dcm2niix_topup(dicom_dir_topup, output_dir_topup, subject_id, session_id)
                
                    # Run cubids commands to add necessary BIDS metadata to top up files.
                    logging.info(f"Adding BIDS metadata to {subject_id}_{session_id}_acq-topup_dir-PA_epi.nii")
                    run_cubids_add_nifti_info(bids_root_dir)
                    json_filepath_topup = os.path.join(output_dir_topup, f'{subject_id}_{session_id}_acq-topup_dir-PA_epi.json')
                    update_json_file_topup(json_filepath_topup, subject_id, session_id, dicom_dir_topup)
                    
                    # Run cubids commands to remove BIDS metadata from top up files.
                    logging.info(f"Removing BIDS metadata from {subject_id}_{session_id}_acq-topup_dir-PA_epi.nii")
                    run_cubids_remove_metadata_fields(bids_root_dir, ['PatientBirthDate'])

                    # Run dcm2niix for verbose output.
                    with tempfile.TemporaryDirectory() as temp_dir:
                        run_dcm2niix_verbose_dwi(dicom_dir_dwi, temp_dir, subject_id, session_id, log_file_path)
                        run_dcm2niix_verbose_topup(dicom_dir_topup, temp_dir, subject_id, session_id, log_file_path)
                
                # Catch error if cubids is not installed.
                else:
                    logging.error("cubids is not installed. Skipping cubids commands.")
                    return # Exit the function if cubids is not installed.
        
        # Catch error if dcm2niix is not installed.
        else:       
            logging.error("dcm2niix is not installed. Cannot proceed with DICOM to NIFTI conversion.")
            return  # Exit the function if dcm2niix is not installed.
        
    # Log other errors. 
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        raise

# Entry point of the script when executed from the command line.
if __name__ == "__main__":
    """
    Entry point of the script when executed from the command line.

    Parses command-line arguments to determine the directories for DICOM files and BIDS dataset.

    Usage:
    process_DKI_to_BIDS.py <dicom_root_dir> <bids_root>
   """
    
    # Set up an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description='Process DICOM files for DKI and convert to NIFTI.')

    # Add arguments to the parser.

    # The first argument is the root directory containing the DICOM directories.
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    
    # The second argument is the root directory of the BIDS dataset.
    parser.add_argument('bids_root_dir', type=str, help='Root directory of the BIDS dataset.')

    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Starting script messages
    print(f"Starting script with provided arguments.")
    print(f"Dicom data directory: {args.dicom_root_dir}")
    print(f"BIDS root directory: {args.bids_root_dir}")

    # Call the main function with the parsed arguments.
    try:
        # Run the main function with the parsed arguments.
        main(args.dicom_root_dir, args.bids_root_dir)
    except Exception as e:
        logging.error("An error occurred during script execution: %s", e, exc_info=True)
        logging.info("Script execution completed with errors.")
    else:
        logging.info("Script executed successfully.")
