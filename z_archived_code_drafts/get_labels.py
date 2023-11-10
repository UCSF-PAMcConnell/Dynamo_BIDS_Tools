import scipy.io

# Replace with the path to your .mat file
mat_file_path = '/Users/PAM201/Documents/MRI/LEARN/BIDS_test/sourcedata/sub-LRN001/ses-1/physio/sub-LRN001_ses-1_task-rest_physio.mat'

# Load the .mat file
mat_contents = scipy.io.loadmat(mat_file_path)

# Print the keys in the .mat file
print("Keys in .mat file:", mat_contents.keys())

# If 'labels' is a key, print its contents
if 'labels' in mat_contents:
    print("Labels:", mat_contents['labels'])

# If the labels are in a nested structure, you might need to adjust the key
# For example, if labels are under 'data' then 'label'
if 'data' in mat_contents and 'label' in mat_contents['data']:
    print("Nested Labels:", mat_contents['data']['label'])