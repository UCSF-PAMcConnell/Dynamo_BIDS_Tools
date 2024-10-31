# Runs the dcm2niix conversion tool to convert DICOM files to NIFTI format.
def run_dcm2niix(input_dir, temp_dir, subject_id, session_id):
    """
    The output files are named according to BIDS (Brain Imaging Data Structure) conventions.

    Parameters:
    - input_dir (str): Directory containing the DICOM files to be converted.
    - output_dir_func (str): Directory where the converted NIFTI files will be saved.
    - subject_id (str): Subject ID, extracted from the DICOM directory path.
    - session_id (str): Session ID, extracted from the DICOM directory path.

    This function uses the dcm2niix tool to convert DICOM files into NIFTI format.
    It saves the output in the specified output directory, structuring the filenames
    according to BIDS conventions. The function assumes that dcm2niix is installed
    and accessible in the system's PATH.

    Usage Example:
    run_dcm2niix('/dicom_root_dir/dicom_sorted/<func_dicoms>', '/bids_root_dir/sub-01/ses-01/func', 'sub-01', 'sub-01', 'ses-01')

    """
    try:
        # Ensure the output directory for functional scans exists.
        os.makedirs(temp_dir, exist_ok=True)
        base_cmd = [
            'dcm2niix',
            '-f', f'{subject_id}_{session_id}_%p', # Naming convention. 
            '-l', 'y', # Losslessly scale 16-bit integers to use maximal dynamic range.
            '-b', 'y', # Save BIDS metadata to .json sidecar. 
            '-p', 'n', # Do not use Use Philips precise float (rather than display) scaling.
            '-x', 'y', # Crop images. This will attempt to remove excess neck from 3D acquisitions.
            '-z', 'n', # Do not compress files.
            '-ba', 'n', # Do not anonymize files (anonymized at MR console). 
            '-i', 'n', # Do not ignore derived, localizer and 2D images. 
            '-m', '2', # Merge slices from same series automatically based on modality. 
            '-o', temp_dir,
            input_dir
        ]

        # Create a temporary directory for the verbose output run.
        with tempfile.TemporaryDirectory() as temp_dir:# Run the actual conversion without verbose output.
            subprocess.run(base_cmd) #capture_output=False, text=False)
        logging.info(f"dcm2niix conversion completed successfully to {temp_dir}.")

    # Log conversion errors.
    except subprocess.CalledProcessError as e:
        logging.error("dcm2niix conversion failed: %s", e)
        raise

    # Log other errors.
    except Exception as e:
        logging.error("An error occurred during dcm2niix conversion: %s", e)
        raise
