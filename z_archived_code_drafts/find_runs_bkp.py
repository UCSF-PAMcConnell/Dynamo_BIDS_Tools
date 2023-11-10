# Identifies runs within the MRI data based on trigger signals and run metadata
def find_runs(data, run_metadata, mri_trigger_data, sampling_rate=5000):
    """
    Parameters:
    - data: The MRI data as a numpy array.
    - run_metadata: A dictionary containing metadata about the run.
    - mri_trigger_data: The MRI trigger channel data as a numpy array.
    - sampling_rate: The sampling rate of the MRI data.
    Returns:
    - A list of dictionaries, each containing a run's data and start index.
    """
    try:
        # Extract run metadata
        repetition_time = run_metadata['RepetitionTime']
        logging.info(f"Repetition time: {repetition_time}")
        num_volumes_per_run = run_metadata['NumVolumes']
        logging.info(f"Number of volumes per run: {num_volumes_per_run}")
        samples_per_volume = int(sampling_rate * repetition_time)
        
        # Verify data integrity: ensure there are enough samples for the expected runs and volumes
        expected_samples = num_volumes_per_run * samples_per_volume
        if len(data) < expected_samples:
            raise ValueError("The data array does not contain enough samples for the expected number of runs and volumes.")

        # Extract trigger points from the MRI trigger data
        trigger_starts = extract_trigger_points(mri_trigger_data)
        
        # Log trigger_starts for debugging purposes
        logging.info(f"Type of trigger_starts from find_runs(): {type(trigger_starts)}")
        logging.info(f"Length of trigger_starts from find_runs(): {len(trigger_starts)}")
        logging.info(f"Shape of trigger_starts from find_runs(): {trigger_starts.shape}")

        # Handle edge case: check if there are enough trigger points for the expected number of runs
        if len(trigger_starts) < num_volumes_per_run:
            raise ValueError("Not enough trigger points for the expected number of runs.")

        runs = []
        current_run = []
        for i in range(len(trigger_starts) - 1):
            if len(current_run) < num_volumes_per_run:
                current_run.append(trigger_starts[i])
            
            if len(current_run) == num_volumes_per_run or trigger_starts[i+1] - trigger_starts[i] > samples_per_volume:
                start_idx = current_run[0]
                end_idx = start_idx + num_volumes_per_run * samples_per_volume

                # Boundary checks: ensure end_idx does not go beyond the length of data
                if end_idx > len(data):
                    logging.warning(f"End index {end_idx} goes beyond the length of the data. Trimming to the data length.")
                    end_idx = len(data)

                segment = data[start_idx:end_idx, :]
                runs.append({'data': segment, 'start_index': start_idx, 'end_index': end_idx})
                logging.info(f"Run found from index {start_idx} to {end_idx}")

                current_run = []

        if len(current_run) == num_volumes_per_run:
            start_idx = current_run[0]
            end_idx = start_idx + num_volumes_per_run * samples_per_volume

            # Boundary checks for the final run
            if end_idx > len(data):
                logging.warning(f"Final run's end index {end_idx} goes beyond the length of the data. Trimming to the data length.")
                end_idx = len(data)

            segment = data[start_idx:end_idx, :]
            runs.append({'data': segment, 'start_index': start_idx, 'end_index': end_idx})
            logging.info(f"Final run found from index {start_idx} to {end_idx}")
        
        return runs
    except KeyError as e:
        logging.error(f"Metadata key error: {e}", exc_info=True)
        raise
    except IndexError as e:
        logging.error(f"Indexing error: {e}", exc_info=True)
        raise
    except ValueError as e:
        logging.error(f"Value error: {e}", exc_info=True)
        raise
    # Catch any other unexpected exceptions
    except Exception as e:
        logging.error("Unexpected error occurred", exc_info=True)
        raise