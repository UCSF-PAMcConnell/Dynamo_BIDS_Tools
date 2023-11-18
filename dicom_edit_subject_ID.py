import pydicom
import os
import argparse

# Rename DICOM subject ID in a folder containing DICOM files.
def rename_dicom_subject_id(folder_path, old_id, new_id):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            dicom = pydicom.dcmread(file_path)
            
            # Assuming 'PatientID' is the field to change. Replace with the correct DICOM tag.
            if dicom.PatientID == old_id:
                dicom.PatientID = new_id
                print(f"Changed subject ID in file: {file_path}")

            if dicom.PatientName == old_id:
                dicom.PatientName = new_id
                print(f"Changed subject name in file: {file_path}")
                
                # Save the modified DICOM file
                dicom.save_as(file_path)
                print(f"Updated file: {file_path}")

def main(folder_path, old_id, new_id):

    rename_dicom_subject_id(folder_path, old_id, new_id)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Rename DICOM subject ID")
    parser.add_argument("folder_path", help="Path to the folder containing DICOM files")
    parser.add_argument("old_id", help="Current incorrect subject ID")
    parser.add_argument("new_id", help="New correct subject ID")

    args = parser.parse_args()

    main(args.folder_path, args.old_id, args.new_id)

