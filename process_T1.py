import os
import argparse
import subprocess
import re
import shutil

def extract_ids(dicom_dir):
    """
    Extracts subject and session IDs from the DICOM directory path.
    
    Parameters:
    dicom_dir (str): Path to the DICOM directory.
    
    Returns:
    tuple: Subject ID and session ID as strings.
    """
    match_subject = re.search(r'sub-(\w+)', dicom_dir)
    match_session = re.search(r'ses-(\w+)', dicom_dir)
    
    if match_subject and match_session:
        return match_subject.group(1), match_session.group(1)
    else:
        raise ValueError("Could not extract subject and session IDs from the directory path.")

def run_dcm2niix(input_dir, output_dir, subject_id, session_id):
    """
    Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format and names the output files according to BIDS conventions.
    
    Parameters:
    input_dir (str): Input directory containing DICOM files.
    output_dir (str): Directory where the conversion results will be saved.
    subject_id (str): Subject ID extracted from the DICOM directory path.
    session_id (str): Session ID extracted from the DICOM directory path.
    """
    dcm2niix_path = os.path.expanduser('~/Documents/MATLAB/software/iNR/BIDS_tools/dcm2niix')
    output_dir_anat = os.path.join(output_dir, 'anat')
    os.makedirs(output_dir_anat, exist_ok=True)
    cmd = [
        dcm2niix_path,
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
        print(f"Cropped file has been renamed to overwrite the original T1w file: {original_file_path}")

# Main code execution starts here
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files and convert them to NIfTI format following BIDS conventions.')
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    args = parser.parse_args()
    
    # Specify the exact directory where the DICOM files are located within the root directory
    dicom_dir = os.path.join(args.dicom_root_dir, 't1_mprage_sag_p2_iso')

    #dicom_dir = args.dicom_root_dir  # Using the provided DICOM root directory directly
    
    subject_id, session_id = extract_ids(dicom_dir)
    
    output_dir = os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}')
    
    run_dcm2niix(dicom_dir, output_dir, subject_id, session_id)
    rename_cropped_file(output_dir, subject_id, session_id)

    # Using the full path of cubids-validate and cubids-add-nifti-info
    pydeface_path = "~/anaconda3/envs/fmri/bin/pydeface"
    pydeface_command = f"python {pydeface_path} {output_dir}/'sub-{subject_id}_ses-{session_id}_T1w'.nii --outfile {output_dir}/'sub-{subject_id}_ses-{session_id}_T1w'.nii --force"
    
    print(f"Executing: {pydeface_command}")
    # Uncomment the following line to actually execute the command
    subprocess.run(pydeface_command, shell=True)