# Main Script

## Purpose and Functionality

The main script (`main.py`) is the entry point for the AI Ticket Processing System. It orchestrates the entire pipeline, from data extraction and preprocessing to model training and evaluation. The main script ensures that all components work together seamlessly to process IT tickets and generate predictions.

## Instructions for Configuring and Running the Main Script

### 1. Configuration

Before running the main script, ensure that the necessary configurations are set up correctly. The main configurations include:

- **MongoDB Configuration:** Update the `config.py` file with your MongoDB URI and database details.
- **Batch Size:** Adjust the `BATCH_SIZE` variable in the main script to control the number of tickets processed in each batch.
- **Cleaned Data File:** Specify the path to the cleaned data file (`CLEANED_DATA_FILE`) in the main script.

### 2. Running the Main Script

To run the main script, follow these steps:

1. **Ensure MongoDB is running:** Make sure that your MongoDB instance is up and running.

2. **Run the main script:**
   ```bash
   python main.py
   ```

### 3. Monitoring the Pipeline

The main script logs the progress and status of the pipeline to a log file (`pipeline.log`). You can monitor the log file to track the progress and identify any issues that may arise during the execution of the pipeline.

### 4. Handling Errors

If any errors occur during the execution of the main script, they will be logged in the log file. Review the log file to identify the cause of the error and take appropriate action to resolve it.

### 5. Customizing the Pipeline

The main script is designed to be flexible and customizable. You can modify the script to suit your specific requirements. Some common customizations include:

- **Adjusting the batch size:** Modify the `BATCH_SIZE` variable to control the number of tickets processed in each batch.
- **Changing the data extraction process:** Update the `fetch_tickets` function in the `data_extraction.py` file to customize the data extraction process.
- **Modifying the data preprocessing steps:** Update the `clean_data` function in the `data_preprocessing.py` file to customize the data preprocessing steps.
- **Updating the model training process:** Modify the `train_and_save_model` function in the `model_training.py` file to customize the model training process.

By following these instructions, you can effectively configure and run the main script to process IT tickets and generate predictions using the AI Ticket Processing System.
