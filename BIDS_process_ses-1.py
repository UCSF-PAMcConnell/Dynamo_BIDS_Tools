"""
BIDS_process_ses_1.py

Description:
This Python script automates the conversion of physiological data from MATLAB format into BIDS-compliant TSV files. 
It is designed for Resting-State fMRI studies, extracting physiological data, such as ECG, respiratory, and EDA signals. 
The script renames channels following BIDS conventions, segments data based on trigger points, and attaches relevant metadata. 
It also includes visualization features, generating plots for quality checks. 
Robust error handling, detailed logging, and a command-line interface are key aspects, ensuring ease of use in BIDS data processing pipelines.

Usage:

Example usage:

python BIDS_process_ses_1.py <dataset_root_dir> <subject_ids> [--pydeface]

Author: PAMcConnell
Created on: 20231113
Last Modified: 20231113
Version: 1.0.0

License:
This software is released under the MIT License.

Dependencies:
- Python 3.12
- Libraries: scipy, pandas, matplotlib, numpy, json, glob
- MATLAB files containing physiological data for Task fMRI studies.

Environment Setup:
- Ensure Python 3.12 and necessary libraries are installed.
- MATLAB files should be accessible in the specified directory, following a structured format.

Change Log:
- 20231113: Initial release with functions for data loading, processing, segmenting, output file writing, and plotting.

Parameters:
dataset_root_dir (str): Root directory of the dataset.
subject_id (str): Identifier for the subject.
session_id (str): Identifier for the session.

"""
import argparse                 # for parsing command-line arguments.
import subprocess               # for executing shell commands.
import os                       # for interacting with the operating system.
import logging                  # for logging to console and file.
import sys                      # for displaying error messages.
import datetime                 # for generating timestamps.
import time                     # for timing script execution.
import re                       # for regular expressions.
import zipfile                  # for uncompressing zip files.
import shutil                   # for moving files.
import glob                     # for finding files.
import tempfile                 # for creating temporary directories.

# Unzip the zip file with subject dicoms. 
import os
import shutil
import tempfile
import zipfile

def unzip_and_move(zip_file_path, sourcedata_dir):
    """
    Unzips a file and moves its contents to a specified directory.

    Args:
    zip_file_path (str): Path to the ZIP file.
    sourcedata_dir (str): Path to the destination directory.

    Example usage:
    zip_file_path = '/path/to/20230712_170324_LRN001_V1.zip'
    sourcedata_dir = '/path/to/destination_directory'
    unzip_and_move(zip_file_path, sourcedata_dir)
    """
    try:
        with tempfile.TemporaryDirectory() as temp_zip_dir:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_zip_dir)

            inner_folder_name = os.path.basename(zip_file_path).replace('.zip', '')
            inner_folder = os.path.join(temp_zip_dir, inner_folder_name)

            # Check and move 'dicom' and 'convert' directories
            for folder_name in ['dicom', 'convert']:
                source_folder = os.path.join(inner_folder, folder_name)
                destination_folder = os.path.join(sourcedata_dir, folder_name)

                if os.path.exists(source_folder):
                    if folder_name == 'dicom':
                        if os.path.exists(destination_folder):
                            shutil.rmtree(destination_folder)
                        shutil.move(source_folder, destination_folder)
                    elif folder_name == 'convert':
                        # Assuming you want to delete the 'convert' directory
                        shutil.rmtree(source_folder)
                else:
                    print(f'No "{folder_name}" directory found inside the ZIP file.')

        print(f'Successfully extracted and moved files from {zip_file_path} to {sourcedata_dir}')
    except Exception as e:
        print(f'Error unzipping the ZIP file: {str(e)}')


# Sets up archival logging for the script, directing log output to both a file and the console.
def setup_logging(subject_id, session_id, dataset_root_dir):
    """
    The function configures logging to capture informational, warning, and error messages. It creates a unique log file for each 
    subject-session combination, stored in a 'logs' directory within the 'doc' folder adjacent to the BIDS root directory.

    Parameters:
    - subject_id (str): The identifier for the subject.
    - session_id (str): The identifier for the session.
    - dataset_root_dir (str): The root directory of the BIDS dataset.

    Returns:
    - log_file_path (str): The path to the log file.
    
    This function sets up a logging system that writes logs to both a file and the console. 
    The log file is named based on the subject ID, session ID, and the script name. 
    It's stored in a 'logs' directory within the 'doc' folder by subject ID, which is located at the same 
    level as the BIDS root directory.

    The logging level is set to INFO, meaning it captures all informational, warning, and error messages.

    Usage Example:
    setup_logging('sub-01', 'ses-1', '/path/to/dataset_root_dir')
    """

    try:
        # Get the current date and time to create a unique timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")   
        
        # Extract the base name of the script without the .py extension.
        script_name = os.path.basename(__file__).replace('.py', '')

        # Construct the log directory path within 'doc/logs'
        log_dir = os.path.join(os.path.dirname(dataset_root_dir), 'doc', 'logs', script_name, timestamp)

        # Create the log directory if it doesn't exist.
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Construct the log file name using timestamp, session ID, and script name.
        log_file_name = f"{script_name}_{timestamp}_{session_id}.log"
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

    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1) # Exiting the script due to logging setup failure.

# Orchestrates MRI data processing and validation in BIDS format.
def main(dataset_root_dir, start_id, end_id, pydeface=False):
    """
    Orchestrates MRI data processing and validation in BIDS format for multiple subjects.

    Parameters:
    dataset_root_dir (str): Path to the dataset root directory.
    subject_ids (list): List of subject IDs.
    pydeface (bool): Flag to indicate whether to apply pydeface on T1 and FLAIR images.

    The function sets up necessary directories, executes a series of data processing 
    commands for each subject, and validates the output against BIDS standards.
    """

    # Define session ID.
    session_id = "ses-1"

    # Extract numerical part from the subject IDs
    start_num = int(re.search(r'\d+', start_id).group())
    end_num = int(re.search(r'\d+', end_id).group())

    # Generate a list of subject IDs based on the provided start and end numbers
    subject_ids = [f"sub-LRN{str(i).zfill(3)}" for i in range(start_num, end_num + 1)]
    try:
        for subject_id in subject_ids:
            sourcedata_dir = os.path.join(dataset_root_dir, 'sourcedata', subject_id, session_id)
            dicom_sorted_dir = os.path.join(dataset_root_dir, subject_id, session_id, 'dicom_sorted')
            dicom_dir = os.path.join(dataset_root_dir, subject_id, session_id, 'dicom')
            print(f"Dicom dir: {dicom_dir}")

            # Define the order and identifiers for different types of runs (multiple methods provided for debugging).
            subject_id_without_prefix = subject_id.replace('sub-', '')  # Remove 'sub-' prefix
            print(f"Subject ID without prefix: {subject_id_without_prefix}")
            session_id_zip = "V1"
            #zip_file_path = os.path.join(sourcedata_dir, f'*_{subject_id_without_prefix}_{session_id_zip}.zip')

            # Construct the pattern for the zip file
            zip_file_pattern = f'{sourcedata_dir}/*_{subject_id_without_prefix}_{session_id_zip}.zip'

            # Use glob to find matching zip files
            zip_files = glob.glob(zip_file_pattern)

            if zip_files:
                zip_file_path = zip_files[0]  # Assuming there's only one matching zip file
                print(f"Zip file path: {zip_file_path}")
                try:
                    # Unzip and move dicom_sorted and dicom
                    unzip_and_move(zip_file_path, sourcedata_dir)
                    print(f'Unzipped and moved files for {subject_id}, session {session_id}')
                except Exception as e:
                    print(f'Error while unzipping and moving files for {subject_id}, session {session_id}: {str(e)}')
            else:
                print(f'ZIP file not found for {subject_id}, session {session_id}')

            # Check if dicom_sorted or dicom directories exist
            if not os.path.exists(dicom_sorted_dir) and not os.path.exists(dicom_dir):
                # Check if the ZIP file exists
                if os.path.exists(zip_file_path):
                    try:
                        # Unzip and move dicom_sorted and dicom
                        unzip_and_move(zip_file_path, sourcedata_dir)
                        print(f'Unzipped and moved files for {subject_id}, session {session_id}')
                    except Exception as e:
                        print(f'Error while unzipping and moving files for {subject_id}, session {session_id}: {str(e)}')
                else:
                    print(f'ZIP file not found for {subject_id}, session {session_id}')
    except Exception as e:
        print(f'Error while processing {subject_id}, session {session_id}: {str(e)}')
        sys.exit(1)

    for subject_id in subject_ids:
        
        # Setup logging, directories, and other pre-processing steps for each subject.
        log_file_path = setup_logging(subject_id, session_id, dataset_root_dir)
        logging.info("Processing subject: %s, session: %s", subject_id, session_id)
        
        # Record the starting time
        start_time = time.time()
        #logging.info(f"Script execution started at: {start_time}")

        # Define root folders for processing.
        sourcedata_root_dir = os.path.join(dataset_root_dir, 'sourcedata', subject_id, session_id)
        behavior_root_dir = os.path.join(sourcedata_root_dir, 'beh', 'preprocessed')
        dicom_root_dir = os.path.join(sourcedata_root_dir, 'dicom_sorted')
        dicom_sort_root_dir = os.path.join(sourcedata_root_dir, 'dicom')
        physio_root_dir = os.path.join(sourcedata_root_dir, 'physio')
        bids_root_dir = os.path.join(dataset_root_dir, 'dataset')
        
        if not os.path.exists(sourcedata_root_dir):
            logging.info(f"Subject directory {sourcedata_root_dir} does not exist. Skipping...")
            continue

        # Define number of resting state runs to process.
        func_rest_extra_arg = " 4"

        # Define processing commands
        commands = [
            "BIDS_sort_dicom_files.py",
            "process_PCASL_to_BIDS.py",
            "process_T1_to_BIDS.py",
            "process_FLAIR_to_BIDS.py",
            "process_task_rest_to_BIDS.py",
            "process_DKI_to_BIDS.py",
            "process_fmap_EPI_to_BIDS.py",
            "process_fmap_gre_to_BIDS.py",
            "BIDS_process_physio_ses_1.py"
        ]

        for command in commands:
            # Construct the base command
            base_command = "python ~/Documents/MATLAB/software/iNR/BIDS_tools/{}"

            # Determine the correct arguments for each command
            
            if command == "BIDS_sort_dicom_files.py":
                cmd = base_command.format(command) + " {} {}".format(physio_root_dir, bids_root_dir)
            elif command == "BIDS_process_physio_ses_1.py":
                cmd = base_command.format(command) + " {} {}".format(physio_root_dir, bids_root_dir)
            elif command == "process_task_rest_to_BIDS.py":
                cmd = base_command.format(command) + " {} {}".format(dicom_root_dir, bids_root_dir) + func_rest_extra_arg
            elif command in ["process_T1_to_BIDS.py", "process_FLAIR_to_BIDS.py"]:
                cmd = base_command.format(command) + " {} {}".format(dicom_root_dir, bids_root_dir)
                if pydeface:
                    cmd += " --pydeface"
            else:
                # For all other commands
                cmd = base_command.format(command) + " {} {}".format(dicom_root_dir, bids_root_dir)

            logging.info(f"Executing: {cmd}")

            try:
                result = subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                logging.error(f"Command '{cmd}' failed with error: {e}")
                # Log the standard error output
                logging.error(f"Command stderr output:\n{result.stderr.decode('utf-8')}")
                logging.error(f"Command stderr output:\n{e.stderr.decode('utf-8')}")
            except Exception as e:
                logging.error(f"An error occurred during subprocess execution: {e}")

    # Change the working directory to dataset_root_dir and execute the cubids-validate command
    try:
        logging.info(f"Changing working directory to: {dataset_root_dir}")
        os.chdir(dataset_root_dir)
        # Now the current working directory is set to the parent of dataset_root_dir
        logging.info(f"Changed working directory to: {os.getcwd()}")

        # Full paths of cubids commands
        cubids_validate_path = "~/anaconda3/envs/fmri/bin/cubids-validate"
        cubids_validate_command = f"python {cubids_validate_path} {bids_root_dir} cubids"
        
        # Execute cubids-validate
        logging.info(f"Executing BIDS validate: {cubids_validate_command}")
        subprocess.run(cubids_validate_command, shell=True, check=True)

    except Exception as e:
        logging.error(f"Error in processing with cubids commands: {e}")

    # Record the ending time
    end_time = time.time()
    #logging.info(f"Script execution ended at: {end_time}")
    
    # Calculate the total run time in seconds
    total_run_time_minutes = (end_time - start_time)/60 

    # Log the total run time
    logging.info(f"Total run time of the script: {total_run_time_minutes:.2f} minutes")

# Main function to run the script from the command line.
if __name__ == "__main__":
    """
    Command-line execution entry point for the script.

    This block allows the script to be run directly from the command line. It uses argparse to handle
    command-line arguments, specifically the subject_id and paths to the directories containing the BIDS dataset. 
    These arguments are then passed to the main function of the script.

   Usage:

    python BIDS_process__ses_1.py <dataset_root_dir> <subject_id>
  
    """   
    # Parse command line arguments
    
    # Set up an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description="Execute processing commands for MRI data to BIDS format conversion.")

    # The first argument is the root directory of the dataset.
    parser.add_argument("dataset_root_dir", help="Path to the root of the dataset.")
    
    # # The second argument is the subject_id.
    # parser.add_argument("subject_ids", nargs='+', help="List of subject IDs.")
   
    # The second argument is the startsubject_id.
    parser.add_argument("--start-id", help="Starting subject ID")
    
    # The third argument is the end subject_id.
    parser.add_argument("--end-id", help="Ending subject ID")

    # The fourth argument is the flag to indicate whether to apply pydeface on T1 and FLAIR images.  
    parser.add_argument("--pydeface", action='store_true', help="Apply pydeface to T1 and FLAIR images.")

    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Process the script with the provided arguments
    print(f"Dataset root directory: {args.dataset_root_dir}")
    print(f"Start ID: {args.start_id}")
    print(f"End ID: {args.end_id}")
    print(f"Pydeface flag: {args.pydeface}")

    # Call the main function with the parsed arguments.
    try:
       main(args.dataset_root_dir, args.start_id, args.end_id, args.pydeface)
    except Exception as e:
        logging.error("An error occurred during script execution: %s", e, exc_info=True)
        logging.info("Script execution completed with errors.")
    else:
        logging.info("Script executed successfully.")