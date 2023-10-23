import os
import argparse
import subprocess
import shutil
import tempfile

def extract_ids(dicom_dir):
    """
    Extracts subject and session IDs from the DICOM directory path.
    
    Parameters:
    dicom_dir (str): Path to the DICOM directory.
    
    Returns:
    tuple: Subject ID and session ID as strings.
    """
    parts = dicom_dir.split('/')
    subject_id = next((part for part in parts if part.startswith('sub-')), None)
    session_id = next((part for part in parts if part.startswith('ses-')), None)
    return subject_id, session_id

def run_dcm2niix(input_dir, output_dir_temp):
    """
    Runs the dcm2niix conversion tool to convert DICOM files to NIfTI format.
    
    Parameters:
    input_dir (str): Input directory containing DICOM files.
    output_dir_temp (str): Temporary directory where the conversion results will be saved.
    """
    cmd = [
        '/Users/PAM201/Documents/MATLAB/software/iNR/BIDS_tools/dcm2niix',
        '-f', '"%p_%s"',
        '-p', 'y',
        '-z', 'n',
        '-ba', 'n',
        '-o', output_dir_temp,
        input_dir
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DKI DICOM files and convert to NIfTI in BIDS format.')
    parser.add_argument('dicom_root_dir', type=str, help='Root DICOM directory containing DKI directories.')
    parser.add_argument('bids_root', type=str, help='Root directory of the BIDS dataset.')
    
    args = parser.parse_args()
    
    # Extract subject and session IDs from the DICOM directory
    subject_id, session_id = extract_ids(args.dicom_root_dir)
    
    dki_dirs = ["DKI_BIPOLAR_2.5mm_64dir_58slices", "DKI_BIPOLAR_2.5mm_64dir_58slices_TOP_UP_PA"]
    
    for dki_dir in dki_dirs:
        dicom_dir = os.path.join(args.dicom_root_dir, dki_dir)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            run_dcm2niix(dicom_dir, tmpdirname)
            
            if "TOP_UP_PA" in dki_dir:
                # Define the fmap directory for top_up files
                output_bids_dir = os.path.join(args.bids_root, subject_id, session_id, 'fmap')
            else:
                # Define the dwi directory for other files
                output_bids_dir = os.path.join(args.bids_root, subject_id, session_id, 'dwi')
                
            os.makedirs(output_bids_dir, exist_ok=True)
            
            for old_file in os.listdir(tmpdirname):
                old_filepath = os.path.join(tmpdirname, old_file)
                
                new_file = f"{subject_id}_{session_id}"
                if "TOP_UP_PA" in dki_dir:
                    new_file += f"_acq-topup_dir-PA_epi{os.path.splitext(old_file)[-1]}"
                else:
                    new_file += f"_dwi{os.path.splitext(old_file)[-1]}"
                
                new_filepath = os.path.join(output_bids_dir, new_file)
                shutil.move(old_filepath, new_filepath)
