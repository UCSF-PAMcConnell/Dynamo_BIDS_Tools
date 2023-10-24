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
    Reads and returns DICOM headers from the specified directory.
    """
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f)) and f.startswith('MR.')]
    dicom_headers = [pydicom.dcmread(df, force=True) for df in dicom_files]
    return dicom_headers

def process_files(output_dirs, subject_id, session_id):
    """
    Renames, organizes, and updates output files in specified directories according to BIDS conventions.
    """
    for output_dir in output_dirs:
        output_dir_perf = os.path.join(output_dir, 'perf')
        
        for file in os.listdir(output_dir_perf):
            old_filepath = os.path.join(output_dir_perf, file)
            
            # Ignoring unexpected or system files
            if file.startswith('.') or not (file.endswith('.nii') or file.endswith('.json')):
                print(f"Warning: Ignoring unexpected file: {file}")
                continue
            
            # Defining the new file name based on BIDS conventions
            prefix = f"sub-{subject_id}_ses-{session_id}"
            if file.endswith('.nii'):
                new_filename = f"{prefix}_asl.nii"
            elif file.endswith('.json'):
                new_filename = f"{prefix}_asl.json"
            
            new_filepath = os.path.join(output_dir_perf, new_filename)
            
            # Renaming and updating the JSON file
            if os.path.exists(old_filepath):
                os.rename(old_filepath, new_filepath)
                if new_filepath.endswith('.json'):
                    update_json_file(new_filepath)
            else:
                print(f"Warning: File not found: {old_filepath}")

def get_num_volumes(dicom_headers):
    """
    Calculates the number of volumes based on the DICOM headers.
    """
    return len(dicom_headers)

def create_aslcontext_file(num_volumes, output_dirs, subject_id, session_id):
    """
    Creates aslcontext.tsv file necessary for BIDS specification.
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
    Updates the JSON sidecar file with necessary fields for BIDS compliance.
    """
    with open(json_filepath, 'r+') as file:
        data = json.load(file)
        data['LabelingDuration'] = 0.7
        data['PostLabelingDelay'] = 1.000
        data['BackgroundSuppression'] = False
        data['M0Type'] = "Absent"
        data['TotalAcquiredPairs'] = 6
        data['VascularCrushing'] = False
        # EffectiveEchoSpacing set to equal DwellTime 
        # (https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/glossary.html#term-Dwell-Time)
        # Need to verify if this needs to be adjusted based on SENSE parallel imaging parameters
        data['EffectiveEchoSpacing'] = 0.0000104
        data['B0FieldSource'] = "*fm2d2r"
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

def extract_ids(dicom_dir):
    """
    Extracts subject and session IDs from the DICOM directory path using regular expressions.
    """
    match_subject = re.search(r'sub-(\w+)', dicom_dir)
    match_session = re.search(r'ses-(\w+)', dicom_dir)

    if match_subject and match_session:
        return match_subject.group(1), match_session.group(1)
    else:
        raise ValueError("Could not extract subject and session IDs from the directory path.")

def run_dcm2niix(input_dir, output_dirs):
    """
    Runs the dcm2niix conversion tool for DICOM to NIfTI conversion.
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files for ASL context and convert to NIfTI.')
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')

    args = parser.parse_args()

    # Specify the exact directory where the DICOM files are located within the root directory
    dicom_dir = os.path.join(args.dicom_root_dir, 'tgse_pcasl_ve11c_from_USC')

    # Extracting subject and session IDs
    subject_id, session_id = extract_ids(args.dicom_root_dir)

    # Specifying output directories
    output_dirs = [
        os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}'),
        args.dicom_root_dir  # Adding the DICOM directory as an output directory
    ]

    # Reading DICOM headers
    dicom_headers = read_dicom_headers(dicom_dir)
    num_volumes = get_num_volumes(dicom_headers)

    # Check if the number of volumes is as expected
    if num_volumes != 12:
        print(f"Warning: Expected 12 volumes but found {num_volumes} volumes.")

    # Creating aslcontext.tsv, converting DICOM to NIfTI, and updating JSON files
    create_aslcontext_file(num_volumes, output_dirs, subject_id, session_id)
    run_dcm2niix(dicom_dir, output_dirs)
    process_files(output_dirs, subject_id, session_id)

    # Copying the 'perf' directory to the DICOM directory
    shutil.copytree(os.path.join(args.bids_root, f'sub-{subject_id}', f'ses-{session_id}', 'perf'),
                    os.path.join(args.dicom_root_dir, 'perf'), dirs_exist_ok=True)
