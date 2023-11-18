#!/bin/zsh learnLoop.sh
# usage: 

# Define an array of subject IDs
subject_ids_ses_1=(
"sub-LRN001"
"sub-LRN002"
"sub-LRN003"
"sub-LRN004"
"sub-LRN005"
"sub-LRN006"
"sub-LRN007"
"sub-LRN008"
"sub-LRN009"
"sub-LRN011"
"sub-LRN012"
"sub-LRN013"
"sub-LRN014"
"sub-LRN015"
"sub-LRN016"
"sub-LRN017"
"sub-LRN019"
)

subject_ids_ses_2=(
"sub-LRN001"
"sub-LRN002"
"sub-LRN003"
"sub-LRN005"
"sub-LRN006"
"sub-LRN007"
"sub-LRN008"
"sub-LRN009"
"sub-LRN011"
"sub-LRN012"
"sub-LRN013"
"sub-LRN014"
"sub-LRN016"
"sub-LRN017"
"sub-LRN019"
)
# Path to your Python script and data directory
script_path="~/Documents/MATLAB/software/iNR/BIDS_tools"
data_dir="~/Documents/MRI/LEARN/BIDS_test/"

# Loop through each subject ID for session 1
for id in "${subject_ids_ses_1[@]}"; do
    python "${script_path}/BIDS_process_ses-1.py" "${data_dir}" --start-id "$id" --end-id "$id"
done

# Loop through each subject ID for session 2 (adjust the list as needed)
for id in "${subject_ids_ses_2[@]}"; do
    python "${script_path}/BIDS_process_ses-2.py" "${data_dir}" --start-id "$id" --end-id "$id"
done
