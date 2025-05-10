import argparse
import os
import json
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_storage_credentials():
    """Get storage credentials from config.json"""
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        
        # Read the config file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract storage credentials
        storage_account_name = config.get('storage_account_name')
        storage_account_key = config.get('storage_account_key')
        storage_connection_string = config.get('storage_connection_string')

        if not all([storage_account_name, storage_account_key, storage_connection_string]):
            raise ValueError("Missing required storage account configuration in config.json")

        return {
            'account_name': storage_account_name,
            'account_key': storage_account_key,
            'connection_string': storage_connection_string
        }

    except FileNotFoundError:
        logger.error("config.json file not found")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config.json")
        raise
    except Exception as e:
        logger.error(f"Error reading config.json: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch data from blob storage')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the fetched data')
    parser.add_argument('--days_back', type=int, default=30, help='Number of days to look back for data')
    return parser.parse_args()

def get_blob_paths_for_date_range(start_date, end_date):
    """Generate blob paths for the given date range"""
    paths = []
    current_date = start_date
    
    while current_date <= end_date:
        # Format: TL01/YYYY/MM/DD/HH
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')
        hour = current_date.strftime('%H')
        
        path = f"TL01/{year}/{month}/{day}/{hour}"
        paths.append(path)
        
        # Move to next hour
        current_date += timedelta(hours=1)
    
    return paths

def fetch_data_from_blob(days_back):
    """Fetch data from blob storage for the specified number of days back"""
    try:
        # Get storage credentials from config.json
        storage_credentials = get_storage_credentials()
        
        # Create the BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(
            storage_credentials['connection_string']
        )
        
        # Get container client
        container_name = os.getenv('BLOB_CONTAINER_NAME', 'iotdata-tl01-iot-app')
        container_client = blob_service_client.get_container_client(container_name)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Fetching data from {start_date} to {end_date}")

        # Get all possible blob paths for the date range
        blob_paths = get_blob_paths_for_date_range(start_date, end_date)
        
        # List blobs in the container
        all_data = []
        for blob in container_client.list_blobs():
            # Check if blob path matches any of our target paths
            if any(blob.name.startswith(path) for path in blob_paths):
                logger.info(f"Processing blob: {blob.name}")
                blob_client = container_client.get_blob_client(blob.name)
                data = blob_client.download_blob().readall()
                
                # Assuming data is in CSV format
                # Modify this based on your actual data format
                df = pd.read_csv(pd.io.common.BytesIO(data))
                all_data.append(df)

        if not all_data:
            raise ValueError(f"No data found for the specified date range")

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data

    except Exception as e:
        logger.error(f"Error fetching data from blob storage: {str(e)}")
        raise

def main():
    args = parse_args()
    
    try:
        # Fetch data from blob storage
        data = fetch_data_from_blob(args.days_back)
        
        # Save the combined data directly to the output path
        data.to_csv(args.output_path, index=False)
        
        logger.info(f"Successfully fetched and saved data to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
