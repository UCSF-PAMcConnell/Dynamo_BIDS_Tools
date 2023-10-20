import pydicom

def extract_dicom_tag(dicom_file_path, tag):
    # Load the DICOM file
    dicom_file = pydicom.dcmread(dicom_file_path)
    
    # Extract the tag value
    tag_value = dicom_file.get(tag)
    
    return tag_value

# Specify the DICOM file path and the tag
dicom_file_path = '/Users/PAM201/Documents/MRI/LEARN/BIDS_test/sourcedata/sub-LRN001/ses-1/dicom_sorted/tgse_pcasl_ve11c_from_USC/MR.1.3.12.2.1107.5.2.43.167021.2023071218025118053156369.dcm'
#tag = (0x0018, 0x9258)
tag = (0x0008, 0x0008)
# Get the tag value
tag_value = extract_dicom_tag(dicom_file_path, tag)

# Print the tag value
print(f"Value of tag {tag}: {tag_value}")
