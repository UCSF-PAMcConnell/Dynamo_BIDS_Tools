import os
import argparse
import subprocess
import shutil
import tempfile
import json

# <dicom_root_dir> <bids_root>

def update_json_file(json_filepath):
    """
    Updates specific fields in a JSON file by adding the 'TaskName' field with the value 'learn',
    and other fields as specified.
    """
    with open(json_filepath, 'r+') as file:
        data = json.load(file)
        data['TaskName'] = 'learn'
        data['B0FieldSource'] = "*fm2d2r"
        data['B0FieldSource2'] = "*epfid2d1_96"
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

def extract_ids(dicom_dir):
    """
    Extracts subject and session IDs from the DICOM directory path.
    """
    parts = dicom_dir.split('/')
    subject_id = next((part for part in parts if part.startswith('sub-')), None)
    session_id = next((part for part in parts if part.startswith('ses-')), None)
    return subject_id, session_id

def run_dcm2niix(input_dir, output_dir_temp):
    """
    Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format.
    """
    dcm2niix_path = os.path.expanduser('~/Documents/MATLAB/software/iNR/BIDS_tools/dcm2niix')
    cmd = [
        dcm2niix_path,
        '-f', '"%p_%s"',
        'l', 'y',
        '-p', 'n',
        '-x', 'n',
        '-z', 'n',
        '-ba', 'n',
        '-o', output_dir_temp,
        input_dir
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files and convert to NIfTI in BIDS format.')
    parser.add_argument('dicom_root_dir', type=str, help='Root directory containing the DICOM directories.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    args = parser.parse_args()
    
    base_dicom_dir = os.path.join(args.dicom_root_dir)
    subject_id, session_id = extract_ids(args.dicom_root_dir)

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

    for suffix, run in run_mapping.items():
        dicom_dir = os.path.join(base_dicom_dir, f'sms3_TASK{suffix}')
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            run_dcm2niix(dicom_dir, tmpdirname)
            
            func_dir_bids = os.path.join(args.bids_root, subject_id, session_id, 'func')
            os.makedirs(func_dir_bids, exist_ok=True)
            
            for old_file in os.listdir(tmpdirname):
                old_filepath = os.path.join(tmpdirname, old_file)
                
                new_file = f"{subject_id}_{session_id}_task-learn_run-{run}_bold"
                new_file += os.path.splitext(old_file)[-1]
                
                new_filepath = os.path.join(func_dir_bids, new_file)
                shutil.move(old_filepath, new_filepath)
                
                if new_filepath.endswith('.json'):
                    update_json_file(new_filepath)
