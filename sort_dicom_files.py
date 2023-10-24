import os
import shutil
import pydicom
import sys

def sort_dicom_files(input_directory, output_directory):
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each entity in the input directory
    for filename in os.listdir(input_directory):
        filepath = os.path.join(input_directory, filename)
        
        # Check if the entity is a file
        if os.path.isfile(filepath):
            # Try to read the DICOM file
            try:
                dicom_file = pydicom.dcmread(filepath)
                
                # Extract sequence information (you might need to adjust this based on your specific DICOM files)
                sequence_name = dicom_file.SeriesDescription  # Adjust attribute if necessary

                # Create a directory for the sequence if it doesn't exist
                sequence_directory = os.path.join(output_directory, sequence_name)
                if not os.path.exists(sequence_directory):
                    os.makedirs(sequence_directory)
                
                # Move the DICOM file to the corresponding sequence folder
                shutil.move(filepath, os.path.join(sequence_directory, filename))
                
            except pydicom.errors.InvalidDicomError:
                print(f"Warning: {filename} is not a valid DICOM file and will be skipped.")
            except Exception as e:
                print(f"Could not process file {filename}: {e}")

# Check if the right number of command-line arguments are provided
if len(sys.argv) != 4:
    print("Usage: python sort_dicom_files.py <sourcedata_root_dir> <subject_id> <session_id>")
    sys.exit(1)

# Get the paths from the command-line arguments
sourcedata_root_dir = sys.argv[1]
subject_id = sys.argv[2]
session_id = sys.argv[3]

# Construct the input and output directories
input_directory = os.path.join(sourcedata_root_dir, subject_id, session_id, 'dicom')
output_directory = os.path.join(sourcedata_root_dir, subject_id, session_id, 'dicom_sorted')

# Call the function with the constructed arguments
sort_dicom_files(input_directory, output_directory)
