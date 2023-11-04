def segment_data_into_runs(data, trigger_starts, tr, sampling_rate, num_volumes):
    """
    Segments the physiological data into runs based on the MRI trigger points.

    Parameters:
    data (np.array): The physiological data array.
    trigger_starts (list): Indices of the start of each MRI trigger point.
    tr (float): The repetition time of the MRI scanner.
    sampling_rate (int): The sampling rate of the physiological data.
    num_volumes (int): The number of volumes (time points) in each fMRI run.

    Returns:
    list: A list of dictionaries, each containing a segmented run and its start index.
    """

    # Initialize a list to hold the segmented runs.
    runs = []
    
    # Initialize variables to track the current run
    current_run = []
    run_start_index = None
    
    # Loop through all the trigger start points to segment the runs
    for i, trigger_index in enumerate(trigger_starts):
        if run_start_index is None:
            # Start a new run if we're not currently tracking one
            run_start_index = trigger_index
            current_run = [trigger_index]
        elif trigger_index - current_run[-1] <= samples_per_volume:
            # Continue the current run if the next trigger is within the expected range
            current_run.append(trigger_index)
        else:
            # End the current run if the gap is too large and start a new one
            runs.append({
                'data': data[run_start_index:current_run[-1] + samples_per_volume, :],
                'start_index': run_start_index
            })
            run_start_index = trigger_index
            current_run = [trigger_index]
        
        # Check if we've reached the expected number of volumes for the run
        if len(current_run) == num_volumes:
            # End the current run and reset for the next run
            runs.append({
                'data': data[run_start_index:trigger_index + samples_per_volume, :],
                'start_index': run_start_index
            })
            run_start_index = None
    
    # Log the total number of runs identified
    logging.info(f'Segmentation complete. Total runs found: {len(runs)}')
    
    # Return the list of segmented runs
    return runs


# Define a function to save the segments into TSV files, excluding unwanted labels
def save_segments_to_tsv(segments, bids_root_dir, subject_id, session_id, original_labels):
    # Map the original labels to the BIDS-compliant labels
    bids_labels = {
        'ECG Test - ECG100C': 'cardiac',
        'RSP Test - RSP100C': 'respiratory',
        'EDA - EDA100C-MRI': 'eda',
        'MRI Trigger - Custom, HLT100C - A 4': 'trigger'
    }
    
    # Identify the indices of columns with labels we want to keep
    columns_to_keep = [idx for idx, label in enumerate(original_labels) if label in bids_labels]
    # Create a new list of labels for the columns we are keeping
    new_labels = [bids_labels[original_labels[idx]] for idx in columns_to_keep]
    
    # Create the output directory if it doesn't exist
    func_dir = os.path.join(bids_root_dir, f"sub-{subject_id}", f"ses-{session_id}", 'func')
    os.makedirs(func_dir, exist_ok=True)
    
    # Log the start of the saving process
    logging.info(f"Saving segments to TSV files in directory: {func_dir}")
    
    for i, segment in enumerate(segments):
        # Create a DataFrame for the current segment with the new labels
        df_segment = pd.DataFrame(segment['data'][:, columns_to_keep], columns=new_labels)
        
        # Prepare the filename and filepath using BIDS naming conventions
        filename = f"sub-{subject_id}_ses-{session_id}_task-rest_run-{str(i+1).zfill(2)}_physio.tsv.gz"
        filepath = os.path.join(func_dir, filename)
        
        # Save the DataFrame as a gzipped TSV file
        df_segment.to_csv(filepath, sep='\t', index=False, compression='gzip')
        
        # Log the successful save action
        logging.info(f"Saved {filename} to {func_dir}")
        
        # Optionally, you can still print to stdout if required
        print(f"Saved {filename} to {func_dir}")

    # Log the completion of the save process
    logging.info("All segments have been saved to TSV files.")

# Define a function to plot the full data with segments highlighted
def plot_full_data_with_segments(data, runs, sampling_rate, original_labels, output_fig_path):
    # Log the start of the plotting process
    logging.info('Starting the plotting of full data with segments.')
    
    # Filter out labels that contain 'Digital input'
    filtered_labels = [label for label in original_labels if 'Digital input' not in label]
    
    # Calculate time points based on the sampling rate and data length
    time = np.arange(data.shape[0]) / sampling_rate
    
    # Determine the number of subplots based on the number of filtered labels
    num_subplots = len(filtered_labels)
    
    # Create subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 10), sharex=True)
    # Ensure axes is a list even if there is only one subplot
    axes = axes if num_subplots > 1 else [axes]
    
    # Suppress any warnings that arise during plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        
        # Plot each label in its own subplot
        for i, label in enumerate(filtered_labels):
            label_index = original_labels.index(label)
            
            # Plot the full data for the current label
            axes[i].plot(time, data[:, label_index], label=f"Full Data {label}", color='lightgrey')
            
            # Overlay run segments for the current label
            for run_idx, run in enumerate(runs):
                # Calculate the time points for the current run segment
                run_time = time[run['start_index']:run['start_index'] + run['data'].shape[0]]
                # Plot the run segment
                axes[i].plot(run_time, run['data'][:, label_index], label=f"Run {run_idx+1} {label}")
            
            # Add the legend to the subplot
            axes[i].legend(loc='upper right')
    
    # Label the x-axis
    plt.xlabel('Time (s)')
    
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure to the specified path
    plt.savefig(output_fig_path, dpi=300)
    
    # Log the completion of the plot and its saving
    logging.info(f'Plot saved to {output_fig_path}')
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Log that the plot is closed
    logging.info('Plotting complete and figure closed.')


# Define the main function to process and convert physiological data to BIDS format
def main(physio_root_dir, bids_root_dir):
    logging.info("Starting the conversion process.")

    # Initialize the list of runs here to ensure it's always defined
    runs = []

    # Extract triggers and segment data into runs
    trigger_label = 'MRI Trigger - Custom, HLT100C - A 4'
    if trigger_label not in labels:
        error_msg = f"Trigger label '{trigger_label}' not found in the labels."
        logging.error(error_msg)
        raise ValueError(error_msg)

    trigger_channel_index = labels.index(trigger_label)
    trigger_starts = extract_trigger_points(data[:, trigger_channel_index])
    logging.info(f"Extracted {len(trigger_starts)} trigger points.")

    # Assume that you know the maximum number of runs you expect
    expected_runs_count = 4  # For example, if you expect 4 runs

    
    # After processing, ensure that runs is defined and has the data
    # Now plot and save the segmented data
    if runs:  # Only attempt to plot if there are runs available
        output_fig_path = mat_file_path.replace('.mat', '.png')
        plot_full_data_with_segments(data, runs, sampling_rate, labels, output_fig_path)
        logging.info(f"Saved plot to {output_fig_path}.")

        # Save the runs as TSV files
        save_segments_to_tsv(runs, bids_root_dir, subject_id, session_id, labels)
        logging.info("All segments saved to TSV files.")
    else:
        logging.info("No runs to process.")

    logging.info("Conversion process completed.")


