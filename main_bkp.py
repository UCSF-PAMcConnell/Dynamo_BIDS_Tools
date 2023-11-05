# Main function to orchestrate the conversion process
def main(physio_root_dir, bids_root_dir):
    logging.info("Starting main processing function.")

    try:
        # Extract subject and session IDs from the path
        subject_id, session_id = extract_subject_session(physio_root_dir)
        logging.info(f"Processing subject: {subject_id}, session: {session_id}")

        # Construct the .mat file path
        mat_file_name = f"{subject_id}_{session_id}_task-rest_physio.mat"
        mat_file_path = os.path.join(physio_root_dir, mat_file_name)

        # Load physiological data from the .mat file
        labels, data, units = load_mat_file(mat_file_path)
        if data is None or not data.size:
            logging.error("Data is empty after loading.")
        else:
            logging.info(f"Data loaded successfully with shape: {data.shape}")
        logging.info("Physiological data loaded successfully.")

        # Rename channels based on BIDS format and create units dictionary
        bids_labels_dictionary, bids_labels_list = rename_channels(labels)
        logging.info(f"BIDS Labels: {bids_labels_list}")        

        # Confirm 'cardiac' is in the BIDS labels list
        if 'cardiac' not in bids_labels_list:
            logging.error("Expected 'cardiac' label is missing from BIDS labels.")
            # Handle the missing label appropriately
        units_dict = {bids_label: unit for label, unit, bids_label in zip(labels, units, bids_labels_list) if label != 'Digital input'}
        logging.info("Channels renamed according to BIDS format and units dictionary created.")

        # Set to keep track of processed JSON files to avoid reprocessing
        processed_jsons = set()
        # List to store data for all runs
        all_runs_data = []
        # List to store metadata for each run
        runs_info = []

        # Find the index of the trigger channel outside the loop
        trigger_original_label = next((orig_label for orig_label, bids_label in bids_labels_dictionary.items() if bids_label == 'trigger'), None)
        if trigger_original_label is None:
            raise ValueError("Trigger label not found in BIDS labels dictionary.")

        trigger_channel_index = np.where(labels == trigger_original_label)[0]
        if trigger_channel_index.size == 0:
            raise ValueError(f"Trigger label '{trigger_original_label}' not found in labels.")
        trigger_channel_index = trigger_channel_index[0]
        # Extract trigger channel data for the current run
        trigger_channel_data = data[:, trigger_channel_index]
        # logging.info(f"Trigger Channel Data: {trigger_channel_data}")

        # Process each run based on BIDS convention
        for run_idx in range(1, 5):  # Assuming 4 runs
            run_id = f"run-{run_idx:02d}"
            json_file_name = f"{subject_id}_{session_id}_task-rest_{run_id}_bold.json"
            json_file_path = os.path.join(bids_root_dir, subject_id, session_id, 'func', json_file_name)
    
            # Extract run metadata from JSON file
            run_metadata = extract_metadata_from_json(json_file_path, processed_jsons)
            logging.info(f"Metadata for run {run_id} extracted successfully.")
            logging.info(f"JSON file path: {json_file_path}")

            # Find the runs in the data using the extracted trigger starts
            current_runs_info = find_runs(data, run_metadata, trigger_channel_data, sampling_rate=5000)     
            if run_metadata is None:
                logging.warning(f"Metadata for run {run_id} could not be found. Skipping.")
                continue
            # Check if any runs were identified
            if not current_runs_info:  # Correct variable to check here is current_runs_info
                logging.error("No runs were identified.")
            else:
                logging.info(f"Runs identified for {run_id} of {len(current_runs_info)} runs")  
            
            for run_info in current_runs_info:
                # Append segmented data for the run
                all_runs_data.append(run_info['data'])
                runs_info.append(run_info)

                # Create metadata dictionary for the current run
                metadata_dict = create_metadata_dict(run_info, 5000, bids_labels_dictionary, bids_labels_list, units_dict)
                
            # Handle output file writing for the current run (function to be implemented)
            output_dir = os.path.join(bids_root_dir, subject_id, session_id, 'func')
            write_output_files(run_info['data'], run_metadata, metadata_dict, labels, output_dir, subject_id, session_id, run_id)
            logging.info(f"Output files for run {run_id} written successfully.")

        # Plot physiological data for all runs
        original_labels = labels  # Assuming 'labels' are the original labels from the .mat file
        sampling_rate = 5000  # Define the sampling rate
        plot_file_path = os.path.join(physio_root_dir, f"{subject_id}_{session_id}_task-rest_all_runs_physio.png")
        
        logging.info(f"Full dataset shape: {data.shape}")
        logging.info(f"Number of runs to plot: {len(runs_info)}")
        logging.info(f"Original labels being passed to plot_runs: {original_labels}")
        logging.info(f"BIDS labels being passed to plot_runs: {bids_labels_list}")

        for idx, run_info in enumerate(runs_info):
            logging.info(f"Run {idx+1} start index: {run_info['start_index']}, data shape: {run_info['data'].shape}")

        # Before plotting, ensure that runs_info contains only unique runs
        unique_runs_info = []
        seen_indices = set()
        for run_info in runs_info:
            start_index = run_info['start_index']
            if start_index not in seen_indices:
                unique_runs_info.append(run_info)
                seen_indices.add(start_index)
            else:
                logging.warning(f"Duplicate run detected at start index {start_index} and will be ignored.")

        runs_info = unique_runs_info
        logging.info(f"Number of unique runs to plot after deduplication: {len(runs_info)}")

        # Call the plot_runs function
        logging.info(f"Runs info before plotting: {runs_info}")
        if not data or not all([run.size > 0 for run in data]):
            raise ValueError("Data list is empty or contains empty arrays.")
        if not runs_info or len(runs_info) == 0:
            raise ValueError("Runs list is empty.")
        plot_runs(data, runs_info, bids_labels_list, sampling_rate, original_labels, plot_file_path)
        logging.info(f"Physiological data plotted and saved to {plot_file_path}.")

        logging.info("Main processing completed without errors.")

    except Exception as e:
        logging.error("An error occurred during the main processing", exc_info=True)
        logging.error("Processing terminated due to an unexpected error.")
        raise