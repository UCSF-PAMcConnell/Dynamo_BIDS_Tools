import os
import argparse
import subprocess
import pydicom
import re
import glob

# ASL volume information taken from https://github.com/neurolabusc/nii_preprocess/blob/master/nii_basil.m
# case SEQUENCE_PCASL_TGSE
        #options.cASL = true;
        #options.groupingOrder = GROUPING_ORDER_REPEATS;
        #options.labelControlPairs = PAIRS_LABEL_THEN_CONTROL;

def read_dicom_headers(dicom_dir):
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) 
                   if os.path.isfile(os.path.join(dicom_dir, f)) and f.startswith('MR.')]
    dicom_headers = [pydicom.dcmread(df, force=True) for df in dicom_files]
    return dicom_headers

def get_num_volumes(dicom_headers):
    return len(dicom_headers)

def create_aslcontext_file(num_volumes, output_dir, subject_id, session_id):
    output_dir_perf = os.path.join(output_dir, 'perf')
    os.makedirs(output_dir_perf, exist_ok=True)
    
    asl_context_filepath = os.path.join(output_dir_perf, f'sub-{subject_id}_ses-{session_id}_aslcontext.tsv')
    with open(asl_context_filepath, 'w') as file:
        file.write('volume_type\n')
        for i in range(num_volumes):
            file.write('LABEL\n' if i % 2 == 0 else 'CONTROL\n')

def rename_output_files(output_dir, subject_id, session_id):
    for ext in ['nii', 'json']:
        old_files = glob.glob(os.path.join(output_dir, 'perf', f'*.{ext}'))
        for old_file in old_files:
            new_file = os.path.join(output_dir, 'perf', f'sub-{subject_id}_ses-{session_id}_asl.{ext}')
            os.rename(old_file, new_file)

def extract_ids(dicom_dir):
    """
    Extract subject and session IDs from the directory path.
    
    Parameters:
    - dicom_dir (str): Directory path of the DICOM files.
    
    Returns:
    - tuple: A tuple containing the subject ID and session ID.
    """
    # Searching for patterns like "sub-XXXX" and "ses-XXXX" in the directory path
    match_subject = re.search(r'sub-(\w+)', dicom_dir)
    match_session = re.search(r'ses-(\w+)', dicom_dir)
    
    if match_subject and match_session:
        return match_subject.group(1), match_session.group(1)
    else:
        raise ValueError("Could not extract subject and session IDs from the directory path.")


def run_dcm2niix(input_dir, output_dir):
    output_dir_perf = os.path.join(output_dir, 'perf')
    os.makedirs(output_dir_perf, exist_ok=True)
    
    cmd = [
        '/Applications/MRIcroGL.app/Contents/Resources/dcm2niix',
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
    parser.add_argument('dicom_dir', type=str, help='Directory containing the DICOM files.')
    parser.add_argument('output_dir', type=str, help='Directory where the output files will be saved.')
    
    args = parser.parse_args()
    
    subject_id, session_id = extract_ids(args.dicom_dir)
    
    dicom_headers = read_dicom_headers(args.dicom_dir)
    num_volumes = get_num_volumes(dicom_headers)
    
    if num_volumes != 12:
        print(f"Warning: Expected 12 volumes but found {num_volumes} volumes.")
    
    create_aslcontext_file(num_volumes, args.output_dir, subject_id, session_id)
    run_dcm2niix(args.dicom_dir, args.output_dir)
    rename_output_files(args.output_dir, subject_id, session_id)
