"""
BIDS_process_ses_2.py

Description:
This Python script automates the conversion of physiological data from MATLAB format into BIDS-compliant TSV files. 
It is designed for Task fMRI studies, extracting physiological data, such as ECG, respiratory, and EDA signals. 
The script renames channels following BIDS conventions, segments data based on trigger points, and attaches relevant metadata. 
It also includes visualization features, generating plots for quality checks. 
Robust error handling, detailed logging, and a command-line interface are key aspects, ensuring ease of use in BIDS data processing pipelines.

Usage:

Example usage:

python BIDS_process_ses_2.py <dataset_root_dir> <subject_ids> [--pydeface]

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
subject_ids (str): Identifier for the subject.
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
import io                       # for text input and output.

# Execute a subprocess and log its output.
def run_and_log_subprocess(command):
    try:
        # Expand the tilde in file paths
        command = [os.path.expanduser(arg) for arg in command]

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # Log the output
        if result.stdout:
            logging.info(f"Subprocess output:\n{result.stdout}")
        if result.returncode != 0 and result.stderr:
            logging.error(f"Subprocess error:\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess '{' '.join(command)}' failed with error: {e}")
        if e.stderr:
            logging.error(f"Subprocess stderr:\n{e.stderr}")
    except Exception as e:
        logging.error(f"An error occurred during subprocess execution: {e}")

# Unzip and move files into 'sourcedata' folder.
def unzip_and_move(zip_file_path, sourcedata_root_dir):
    """
    Unzips a file and moves its contents to a specified directory.

    Args:
    zip_file_path (str): Path to the ZIP file.
    sourcedata_root_dir (str): Path to the destination directory.

    Example usage:
    zip_file_path = '/path/to/20230712_170324_LRN001_V2.zip'
    sourcedata_root_dir = '/path/to/destination_directory'
    unzip_and_move(zip_file_path, sourcedata_root_dir)
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
                destination_folder = os.path.join(sourcedata_root_dir, folder_name)

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

        print(f'Successfully extracted and moved files from {zip_file_path} to {sourcedata_root_dir}')
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
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H")   
        
        # Extract the base name of the script without the .py extension.
        script_name = os.path.basename(__file__).replace('.py', '')

        # Construct the log directory path within 'doc/logs'
        log_dir = os.path.join(os.path.dirname(dataset_root_dir), 'doc', 'logs', script_name, timestamp)

        # Create the log directory if it doesn't exist.
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Construct the log file name using timestamp, session ID, and script name.
        log_file_name = f"{script_name}_{timestamp}.log"
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

    # Record the starting time for the entire script
    script_start_time = time.time()

    # Define session ID.
    session_id = "ses-2"

    # Extract numerical part from the subject IDs
    start_num = int(re.search(r'\d+', start_id).group())
    end_num = int(re.search(r'\d+', end_id).group())

    # Generate a list of subject IDs based on the provided start and end numbers
    subject_ids = [f"sub-LRN{str(i).zfill(3)}" for i in range(start_num, end_num + 1)]

    for subject_id in subject_ids:
        try:
            # Record the start time for this subject
            subject_start_time = time.time()
            sourcedata_root_dir = os.path.join(dataset_root_dir, 'sourcedata', subject_id, session_id)
            dicom_sorted_dir = os.path.join(dataset_root_dir, subject_id, session_id, 'dicom_sorted')
            dicom_dir = os.path.join(dataset_root_dir, subject_id, session_id, 'dicom')
            print(f"Dicom dir: {dicom_dir}")

            if os.path.exists(dicom_dir):
                print(f"Dicom directory {dicom_dir} already exists. Skipping.")
                continue

            # Define the order and identifiers for different types of runs (multiple methods provided for debugging).
            subject_id_without_prefix = subject_id.replace('sub-', '')  # Remove 'sub-' prefix
            session_id_zip = "V2"
            zip_file_pattern = f'{sourcedata_root_dir}/*_{subject_id_without_prefix}_{session_id_zip}.zip'
            zip_files = glob.glob(zip_file_pattern)

            if zip_files:
                zip_file_path = zip_files[0]
                print(f"Zip file path: {zip_file_path}")

                try:
                    # Unzip and move dicom_sorted and dicom
                    unzip_and_move(zip_file_path, sourcedata_root_dir)
                    print(f'Unzipped and moved files for {subject_id}, session {session_id}')
                except Exception as e:
                    print(f'Error while unzipping and moving files for {subject_id}, session {session_id}: {str(e)}')
            else:
                print(f'ZIP file not found for {subject_id}, session {session_id}')
        
            # Setup logging, directories, and other pre-processing steps for each subject.
            log_dir = setup_logging(subject_id, session_id, dataset_root_dir)
            logging.info("Processing subject: %s, session: %s", subject_id, session_id)

            # Define root folders for processing.
            sourcedata_root_dir = os.path.join(dataset_root_dir, 'sourcedata', subject_id, session_id)
            behavior_root_dir = os.path.join(sourcedata_root_dir, 'beh', 'preprocessed')
            dicom_root_dir = os.path.join(sourcedata_root_dir, 'dicom_sorted')
            physio_root_dir = os.path.join(sourcedata_root_dir, 'physio')
            bids_root_dir = os.path.join(dataset_root_dir, 'dataset')

            if not os.path.exists(sourcedata_root_dir):
                logging.info(f"Subject directory {sourcedata_root_dir} does not exist. Skipping...")
                continue

            # Define processing commands
            commands = [
                "BIDS_sort_dicom_files.py",
                "process_T1_to_BIDS.py",
                "process_task_learn_to_BIDS.py",
                "process_fmap_EPI_to_BIDS.py",
                "process_fmap_gre_to_BIDS.py",
                "process_task_learn_beh_to_BIDS.py",
                "BIDS_process_physio_ses_2.py"
            ]
            
            for command in commands:
                # Define the path to the script, expanding the user directory
                script_path = os.path.expanduser("~/Documents/MATLAB/software/iNR/BIDS_tools/" + command)

                # Start with the base command
                cmd = ["python", script_path]

                # Add arguments based on the command
                if command == "BIDS_sort_dicom_files.py":
                    cmd.extend([sourcedata_root_dir, bids_root_dir])
                elif command == "BIDS_process_physio_ses_2.py":
                    cmd.extend([physio_root_dir, bids_root_dir])
                elif command == "process_task_learn_to_BIDS.py":
                    cmd.extend([dicom_root_dir, bids_root_dir])
                elif command == "process_task_learn_beh_to_BIDS.py":
                    cmd.extend([behavior_root_dir, bids_root_dir])
                elif command in ["process_T1_to_BIDS.py"]:
                    cmd.extend([dicom_root_dir, bids_root_dir])
                    if pydeface:
                        cmd.append("--pydeface")
                else:
                    # For all other commands
                    cmd.extend([dicom_root_dir, bids_root_dir])
                
                # Log the command being executed
                logging.info(f"Executing: {' '.join(cmd)}")

                # Execute the subprocess command
                run_and_log_subprocess(cmd)
        
        except Exception as e:
            print(f'Error while processing {subject_id}, session {session_id}: {str(e)}')
            sys.exit(1)
        
        # Record the end time for this subject
        subject_end_time = time.time()

        # Calculate the time taken for this subject
        subject_elapsed_time = (subject_end_time - subject_start_time) / 60  # Convert to minutes
        logging.info(f"Time taken for subject {subject_id}: {subject_elapsed_time:.2f} minutes")

               
    # Change the working directory to dataset_root_dir and execute the cubids-validate command
    try:
        #logging.info(f"Changing working directory to: {dataset_root_dir}")
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

    # Record the end time for the entire script
    script_end_time = time.time()

    # Calculate the total run time for the script
    total_script_time = (script_end_time - script_start_time) / 60  # Convert to minutes
    logging.info(f"Total run time of the script: {total_script_time:.2f} minutes")
    
# Main function to run the script from the command line.
if __name__ == "__main__":
    """
    Command-line execution entry point for the script.

    This block allows the script to be run directly from the command line. It uses argparse to handle
    command-line arguments, specifically the subject_id and paths to the directories containing the BIDS dataset. 
    These arguments are then passed to the main function of the script.

   Usage:

    python BIDS_process_ses_2.py <dataset_root_dir> <subject_ids> [--pydeface]
  
    """   
    # Parse command line arguments
    
    # Set up an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(description="Execute processing commands for MRI data to BIDS format conversion.")

    # The first argument is the root directory of the dataset.
    parser.add_argument("dataset_root_dir", help="Path to the root of the dataset.")
    
    # The second argument is the startsubject_id.
    parser.add_argument("--start-id", help="Starting subject ID")
    
    # The third argument is the end subject_id.
    parser.add_argument("--end-id", help="Ending subject ID")

    # The third argument is the flag to indicate whether to apply pydeface on T1 and FLAIR images.  
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