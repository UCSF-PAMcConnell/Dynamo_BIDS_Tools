import argparse
import subprocess
import os
import logging
import sys

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Returns the name of the current conda environment.
def get_conda_env():
    try:
        env_name = os.environ.get('CONDA_DEFAULT_ENV')
        return env_name
    except Exception as e:
        logging.error(f"Error getting Conda environment: {e}")
        return None

# Executes a series of processing commands for MRI data.
def execute_commands(sourcedata_root_dir, subject_id, session_id):
    """
    Parameters:
    sourcedata_root_dir (str): Root directory of the source data.
    subject_id (str): Identifier for the subject.
    session_id (str): Identifier for the session.
    """
    # List of processing commands
    commands = [
        "process_PCASL.py",
        "process_T1.py",
        "process_T2_FLAIR.py",
        "process_func_rest.py",
        "process_dki.py",
        "process_fmap_gre_ses_1.py",
        "process_fmap_EPI_ses_1.py"
    ]
    
    # Base command pattern
    base_command = "python ~/Documents/MATLAB/software/iNR/BIDS_tools/{} {}{}/{}/dicom_sorted/ ~/Documents/MRI/LEARN/BIDS_test/dataset"
    func_rest_extra_arg = " 4"
    
    # Determine bids_root_dir
    bids_root_dir = os.path.join(os.path.dirname(os.path.dirname(sourcedata_root_dir)), 'dataset')

    # Ensure the directory exists
    dir_to_create = os.path.join(bids_root_dir, subject_id, session_id)
    os.makedirs(dir_to_create, exist_ok=True)
    
    for command in commands:
        # Construct the command
        cmd = base_command.format(command, sourcedata_root_dir, subject_id, session_id)
        if command == "process_func_rest.py":
            cmd += func_rest_extra_arg
        
        logging.info(f"Executing: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Command '{cmd}' failed with error: {e}")

    # Change the working directory to bids_root_dir and execute the cubids-validate command
    try:
        os.chdir(bids_root_dir)
        logging.info(f"Changed working directory to: {os.getcwd()}")

        # Full paths of cubids commands
        cubids_add_nii_hdr_path = "~/anaconda3/envs/fmri/bin/cubids-add-nifti-info"
        cubids_add_nii_hdr_command = f"python {cubids_add_nii_hdr_path} {bids_root_dir}"
        cubids_validate_path = "~/anaconda3/envs/fmri/bin/cubids-validate"
        cubids_validate_command = f"python {cubids_validate_path} {bids_root_dir} cubids"
        
        # Execute cubids-add-nifti-info
        logging.info(f"Executing: {cubids_add_nii_hdr_command}")
        subprocess.run(cubids_add_nii_hdr_command, shell=True, check=True)

        # Execute cubids-validate
        logging.info(f"Executing: {cubids_validate_command}")
        subprocess.run(cubids_validate_command, shell=True, check=True)

    except Exception as e:
        logging.error(f"Error in processing with cubids commands: {e}")

if __name__ == "__main__":
    
    # Desired Conda environment name
    desired_env = "datalad"

    # Check if the correct Conda environment is activated
    current_env = get_conda_env()
    if current_env != desired_env:
        logging.error(f"Script is not running in the '{desired_env}' Conda environment. Current environment: '{current_env}'")
        logging.info(f"Please activate the correct Conda environment by running: conda activate {desired_env}")
        sys.exit(1)
    else:
        logging.info(f"Running in the correct Conda environment: '{current_env}'")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Execute processing commands for MRI data.')
    parser.add_argument('sourcedata_root_dir', type=str, help='Path to the sourcedata root directory.')
    parser.add_argument('subject_id', type=str, help='Subject ID.')
    parser.add_argument('session_id', type=str, help='Session ID.')
    
    args = parser.parse_args()
    
    # Execute the processing commands
    execute_commands(args.sourcedata_root_dir, args.subject_id, args.session_id)
