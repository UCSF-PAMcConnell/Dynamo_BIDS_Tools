## BIDS_tools
 code for BIDS data processing / inputs: dicom_root_dir or physio_root_dir, bids_root_dir
 
 assumes BIDS compliant directory structure (where 'BIDS' is project/dataset root directory):

 /BIDS/code
 /BIDS/doc
 /BIDS/doc/logs
 /BIDS/derivatives
 /BIDS/sourcedata/ (dicom_root_dir - dicom zip archive path)) + /sub-/ses-/dicom -> dicom_sorted
 /BIDS/sourcedata/sub-/ses-/physio (physio_root_dir)
 /BIDS/sourcedata/sub-/ses-/beh (beh_root_dir)
 /BIDS/dataset (bids_root_dir)
 
## usage

# Sort Dicoms - ses-1

python ~/Documents/MATLAB/software/iNR/BIDS_tools/BIDS_sort_dicom_files.py ~/Documents/MRI/LEARN/BIDS_test/sourcedata/sub-LRN001/ses-1

# Process dicom to nifti BIDs ses-1

python ~/Documents/MATLAB/software/iNR/BIDS_tools/BIDS_process_ses-1.py ~/Documents/MRI/LEARN/BIDS_test/sourcedata/ --start-id sub-LRN001 --end-id sub-LRN00x [--pydeface]

# Sort Dicoms - ses-2

python ~/Documents/MATLAB/software/iNR/BIDS_tools/sort_dicom_files.py ~/Documents/MRI/LEARN/BIDS_test/sourcedata/sub-LRN001/ses-2

# Process dicom to nifti BIDs ses-2

python ~/Documents/MATLAB/software/iNR/BIDS_tools/BIDS_process_ses-2.py ~/Documents/MRI/LEARN/BIDS_test/sourcedata/ --start-id sub-LRN001 --end-id sub-LRN00x [--pydeface]

# Process physio to BIDS - ses-1

python ~/Documents/MATLAB/software/iNR/BIDS_tools/BIDS_process_physio_ses_1.py ~/Documents/MRI/LEARN/BIDS_test/sourcedata/sub-LRN001/ses-1/physio/  ~/Documents/MRI/LEARN/BIDS_test/dataset [--force]

# Process physio to BIDS - ses-2

python ~/Documents/MATLAB/software/iNR/BIDS_tools/BIDS_process_physio_ses_2.py ~/Documents/MRI/LEARN/BIDS_test/sourcedata/sub-LRN001/ses-2/physio/  ~/Documents/MRI/LEARN/BIDS_test/dataset [--force]
