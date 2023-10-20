import os
import argparse
import subprocess
import pydicom
import re
import glob
import json
import shutil

# ASL volume information taken from https://github.com/neurolabusc/nii_preprocess/blob/master/nii_basil.m
# case SEQUENCE_PCASL_TGSE
        #options.cASL = true;
        #options.groupingOrder = GROUPING_ORDER_REPEATS;
        #options.labelControlPairs = PAIRS_LABEL_THEN_CONTROL;
# I'm not 100% on the ASL metadata defined below, needs verification - PAMcConnell 20231020
# usage example: python ~/Documents/MATLAB/software/iNR/BIDS_tools/process_PCASL.py 
# ~/Documents/MRI/LEARN/BIDS_test/sourcedata/sub-LRN001/ses-1/dicom_sorted/tgse_pcasl_ve11c_from_USC/ <dicom_dir>
# ~/Documents/MRI/LEARN/BIDS_test/dataset <bids_root>

def read_dicom_headers(dicom_dir):
    """
    Reads DICOM headers from DICOM files in the specified directory.
    
    Parameters:
    dicom_dir (str): Path to the directory containing DICOM files.
    
    Returns:
    list: A list of pydicom Dataset objects representing DICOM headers.
    """
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) 
                   if os.path.isfile(os.path.join(dicom_dir, f)) and f.startswith('MR.')]
    dicom_headers = [pydicom.dcmread(df, force=True) for df in dicom_files]
    return dicom_headers

def get_num_volumes(dicom_headers):
    """
    Gets the number of volumes (DICOM files) from the DICOM headers.
    
    Parameters:
    dicom_headers (list): A list of pydicom Dataset objects representing DICOM headers.
    
    Returns:
    int: Number of volumes.
    """
    return len(dicom_headers)

def create_aslcontext_file(num_volumes, output_dirs, subject_id, session_id):
    """
    Creates aslcontext.tsv files in specified output directories. 
    
    Parameters:
    num_volumes (int): Number of volumes.
    output_dirs (list): List of directories where the output files will be saved.
    subject_id (str): Subject ID extracted from the DICOM directory path.
    session_id (str): Session ID extracted from the DICOM directory path.
    """
    for output_dir in output_dirs:
        output_dir_perf = os.path.join(output_dir, 'perf')
        os.makedirs(output_dir_perf, exist_ok=True)
        
        asl_context_filepath = os.path.join(output_dir_perf, f'sub-{subject_id}_ses-{session_id}_aslcontext.tsv')
        with open(asl_context_filepath, 'w') as file:
            file.write('volume_type\n')
            for i in range(num_volumes):
                file.write('label\n' if i % 2 == 0 else 'control\n')

def update_json_file(json_filepath):
    """
    Updates specific fields in a JSON file.
    
    Parameters:
    json_filepath (str): Path to the JSON file to be updated.
    """
    with open(json_filepath, 'r+') as file:
        data = json.load(file)
        data['LabelingDuration'] = 0.7
        data['PostLabelingDelay'] = 1.000
        data['BackgroundSuppression'] = False
        data['M0Type'] = "Absent"
        data['TotalAcquiredPairs'] = 6  
        data['VascularCrushing'] = False
        data['LabelingDuration'] = 0.7
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

def rename_output_files(output_dirs, subject_id, session_id):
    """
    Renames and updates output files in specified directories.
    
    Parameters:
    output_dirs (list): List of directories containing the output files to be renamed and updated.
    subject_id (str): Subject ID extracted from the DICOM directory path.
    session_id (str): Session ID extracted from the DICOM directory path.
    """
    for output_dir in output_dirs:
        for ext in ['nii', 'json']:
            old_files = glob.glob(os.path.join(output_dir, 'perf', f'*.{ext}'))
            for old_file in old_files:
                new_file = os.path.join(output_dir, 'perf', f'sub-{subject_id}_ses-{session_id}_asl.{ext}')
                os.rename(old_file, new_file)
                
                if ext == 'json':
                    update_json_file(new_file)

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

def run_dcm2niix(input_dir, output_dirs):
    """
    Runs the dcm2niix conversion for each directory in output_dirs.
    
    Parameters:
    input_dir (str): Input directory containing DICOM files.
    output_dirs (list): List of output directories where the conversion results will be saved.
    """
    for output_dir in output_dirs:
        output_dir_perf = os.path.join(output_dir, 'perf')
        os.makedirs(output_dir_perf, exist_ok=True)
        
        cmd = [
            '/Users/PAM201/Documents/MATLAB/software/iNR/BIDS_tools/dcm2niix',
            '-f', '"sub-%i_%p"',
            '-p', 'y',
            '-z', 'n',
            '-ba', 'n',
            '-o', output_dir_perf,
            input_dir
        ]
        subprocess.run(cmd)

# Main code execution starts here when the script is run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files for ASL context and convert to NIfTI.')
    parser.add_argument('dicom_dir', type=str, help='Directory containing the DICOM files.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    args = parser.parse_args()
    
    subject_id, session_id = extract_ids(args.dicom_dir)
    
    output_dirs = [
        os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}'),
        args.dicom_dir  # Adding the DICOM directory as an output directory
    ]
    
    dicom_headers = read_dicom_headers(args.dicom_dir)
    num_volumes = get_num_volumes(dicom_headers)
    
    if num_volumes != 12:
        print(f"Warning: Expected 12 volumes but found {num_volumes} volumes.")
    
    create_aslcontext_file(num_volumes, output_dirs, subject_id, session_id)
    run_dcm2niix(args.dicom_dir, output_dirs)
    rename_output_files(output_dirs, subject_id, session_id)

    # Copy the 'perf' directory from the BIDS directory to the DICOM directory
    shutil.copytree(os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}', 'perf'), 
                    os.path.join(args.dicom_dir, 'perf'), dirs_exist_ok=True)
