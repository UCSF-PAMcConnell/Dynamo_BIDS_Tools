"""
BIDS_sort_dicom_files.py

Description:
This script sorts DICOM files from an MRI session into separate directories based on their series description, 
facilitating organization according to the Brain Imaging Data Structure (BIDS) standard. 
It reads each DICOM file, extracts its series description, and moves it into a corresponding 
subdirectory within the output directory. The script is particularly useful for preparing 
MRI data for BIDS-compliant processing pipelines.

Usage:
python BIDS_sort_dicom_files.py <sourcedata_root_dir> <bids_root_dir>
e.g.,
/dataset_root_dir/sourcedata_root_dir/sub-01/ses-01/dicom # <sourcedata_root_dir>
/dataset_root_dir/bids_root_dir/ <bids_root_dir>

Author: PAMcConnell
Created on: 20231111
Last Modified: 20231111

License: MIT License

Dependencies:
- Python 3.12
- pydicom
- os, shutil, sys (standard Python libraries)

Environment Setup:
Ensure Python 3.12 and pydicom library are installed in your environment.
You can install pydicom using conda: `conda install -c conda-forge pydicom`.
To set up the required environment, you can use the provided environment.yml file with Conda. <datalad.yml>

Change Log:
- 20231111: Initial version

"""

import os                     # Used for operating system dependent functionalities like file path manipulation.
import shutil                 # Provides high-level file operations like file copying and removal.
import pydicom                # Library for working with DICOM (Digital Imaging and Communications in Medicine) files.
import sys                    # Provides access to some variables used or maintained by the interpreter.
import logging                # Logging library, for tracking events that happen when running the software.
import argparse               # Parser for command-line options, arguments, and sub-commands.
import re                     # Regular expressions, useful for text matching and manipulation.

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
def extract_subject_session(sourcedata_root_dir):
    """
    Parameters:
    - sourcedata_root_dir (str): The directory path that includes subject and session information. 
                             This path should follow the BIDS convention, containing 'sub-' and 'ses-' prefixes.

    Returns:
    - subject_id (str): The extracted subject ID.
    - session_id (str): The extracted session ID.

    Raises:
    - ValueError: If the subject_id and session_id cannot be extracted from the path.

    This function assumes that the directory path follows the Brain Imaging Data Structure (BIDS) naming convention. 
    It uses regular expressions to find and extract the subject and session IDs from the path.

    Usage Example:
    subject_id, session_id = extract_subject_session('/path/to/data/sub-01/ses-1/dicom')

    Note: This function will raise an error if it cannot find a pattern matching the BIDS convention in the path.
    """

    # Normalize the path to remove any trailing slashes for consistency.
    sourcedata_root_dir = os.path.normpath(sourcedata_root_dir)

    # The pattern looks for 'sub-' followed by any characters until a slash, and similar for 'ses-'.
    match = re.search(r'(sub-[^/]+)/(ses-[^/]+)', sourcedata_root_dir)
    if not match:
        raise ValueError("Unable to extract subject_id and session_id from path: %s", sourcedata_root_dir)
    
    subject_id, session_id = match.groups()

    return subject_id, session_id

# Sorts DICOM files into separate directories based on their series description.
def sort_dicom_files(input_directory, output_directory):
    """
    Parameters:
    - input_directory (str): Path to the directory containing DICOM files.
    - output_directory (str): Path to the directory where sorted DICOM files will be stored.

    The function reads each DICOM file, extracts its series description, and moves it into
    a corresponding subdirectory within the output directory.
    """
    # Ensure output directory exists.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each entity in the input directory.
    for filename in os.listdir(input_directory):
        filepath = os.path.join(input_directory, filename)
        
        # Check if the entity is a file.
        if os.path.isfile(filepath):
            # Try to read the DICOM file.
            try:
                dicom_file = pydicom.dcmread(filepath)
                
                # Extract sequence information (you might need to adjust this based on your specific DICOM files).
                sequence_name = dicom_file.SeriesDescription  # Adjust attribute if necessary.

                # Create a directory for the sequence if it doesn't exist.
                sequence_directory = os.path.join(output_directory, sequence_name)
                if not os.path.exists(sequence_directory):
                    os.makedirs(sequence_directory)
                
                output_filepath = os.path.join(sequence_directory, filename)
                logging.info(f"Moving {filename} to {sequence_directory}")

                if not os.path.exists(sequence_directory):
                    os.makedirs(sequence_directory)
                
                if os.path.exists(output_filepath):
                    logging.warning(f"File {filename} already exists in {output_filepath}. Skipping.")
                    continue  # Skip this file as it's already processed.

                # Move the file to the output directory.
                shutil.move(filepath, output_filepath)

            # Handle exceptions for invalid DICOM files.
            except pydicom.errors.InvalidDicomError:
                logging.warning(f"{filename} is not a valid DICOM file and will be skipped.")
            
            # Handle exceptions for other errors.
            except Exception as e:
                logging.error(f"Could not process file {filename}: {e}")

# Main function orchestrating the conversion of DICOM files to BIDS format.
def main(sourcedata_root_dir, bids_root_dir):   

    # Setup logging after extracting subject_id and session_id.
    subject_id, session_id = extract_subject_session(sourcedata_root_dir)
    
    # Setup the input and output directories.
    input_directory = os.path.join(sourcedata_root_dir, 'dicom')
    output_directory = os.path.join(sourcedata_root_dir, 'dicom_sorted')

    try: 
        if os.path.exists(output_directory):
            print(f"Output directory {output_directory} already exists. Skipping.")
            return # Skip if output directory already exists
    except:
        print(f"Output directory {output_directory} exists. Exiting script.")
        sys.exit(1)

    setup_logging(subject_id, session_id, bids_root_dir)
    logging.info(f"Sorting dicoms for  subject: {subject_id}, session: {session_id}")
    
    sort_dicom_files(input_directory, output_directory)
    shutil.rmtree(input_directory)

# Main function to run the script from the command line.
if __name__ == '__main__':
    """
    Entry point for the script when run from the command line.
    Uses argparse to parse command-line arguments and passes them to the main function.
    """
    
    # Create an argument parser.
    parser = argparse.ArgumentParser(description="Sort DICOM files for conversion to BIDS format.")
    parser.add_argument("sourcedata_root_dir", help="Directory containing the dicom file folder for the session.")
    parser.add_argument("bids_root_dir", help="Path to the root of the BIDS dataset.")
    args = parser.parse_args()
    
    # Execute the main function.
    main(args.sourcedata_root_dir, args.bids_root_dir)