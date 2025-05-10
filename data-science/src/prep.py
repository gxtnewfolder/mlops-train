import argparse

from pathlib import Path
import os
import numpy as np
import pandas as pd

import mlflow

TARGET_COL = "I_f"

NUMERIC_COLS = [
    "I_y",
    "PF",
    "e_PF",
    "d_if",
]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--val_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")

    parser.add_argument("--enable_monitoring", type=str, help="enable logging to ADX")
    parser.add_argument("--table_name", type=str, default="mlmonitoring", help="Table name in ADX for logging")

    args = parser.parse_args()

    return args

def log_training_data(df, table_name):
    from obs.collector import Online_Collector
    collector = Online_Collector(table_name)
    collector.batch_collect(df)
    
def main(args):
    '''Read, split, and save datasets'''
    try:
        # ------------ Reading Data ------------ #
        # -------------------------------------- #
        print(f"Reading data from {args.raw_data}")
        data = pd.read_csv((Path(args.raw_data)))
        
        # Validate that required columns exist
        missing_cols = [col for col in NUMERIC_COLS + [TARGET_COL] if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select only the columns we need
        data = data[NUMERIC_COLS + [TARGET_COL]]
        
        # Log data quality metrics
        mlflow.log_metric('raw_data_rows', data.shape[0])
        mlflow.log_metric('missing_values', data.isna().sum().sum())
        
        # Check for missing values
        if data.isna().any().any():
            print("Warning: Dataset contains missing values")
            mlflow.log_metric('has_missing_values', 1)

        # ------------- Split Data ------------- #
        # -------------------------------------- #

        # Split data into train, val and test datasets with fixed random seed for reproducibility
        np.random.seed(42)  # Set seed for reproducibility
        random_data = np.random.rand(len(data))

        msk_train = random_data < 0.7
        msk_val = (random_data >= 0.7) & (random_data < 0.85)
        msk_test = random_data >= 0.85

        train = data[msk_train]
        val = data[msk_val]
        test = data[msk_test]

        # Log dataset sizes
        mlflow.log_metric('train size', train.shape[0])
        mlflow.log_metric('val size', val.shape[0])
        mlflow.log_metric('test size', test.shape[0])
        
        # Log data split percentages
        mlflow.log_metric('train_percentage', train.shape[0] / len(data))
        mlflow.log_metric('val_percentage', val.shape[0] / len(data))
        mlflow.log_metric('test_percentage', test.shape[0] / len(data))

        # Save datasets to parquet files
        try:
            train.to_parquet((Path(args.train_data) / "train.parquet"))
            val.to_parquet((Path(args.val_data) / "val.parquet"))
            test.to_parquet((Path(args.test_data) / "test.parquet"))
            print(f"Data successfully saved to {args.train_data}, {args.val_data}, and {args.test_data}")
        except Exception as e:
            print(f"Error saving datasets: {e}")
            mlflow.log_metric("data_save_error", 1)
            raise

        # Log data to monitoring service if enabled
        if args.enable_monitoring and (args.enable_monitoring.lower() == 'true' or 
                                      args.enable_monitoring == '1' or 
                                      args.enable_monitoring.lower() == 'yes'):
            try:
                print(f"Logging data to monitoring table: {args.table_name}")
                log_training_data(data, args.table_name)
                mlflow.log_metric('monitoring_enabled', 1)
            except Exception as e:
                print(f"Warning: Failed to log data to monitoring service: {e}")
                mlflow.log_metric('monitoring_error', 1)
                
    except Exception as e:
        print(f"Error in data preparation: {e}")
        mlflow.log_metric('prep_error', 1)
        raise



if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Val dataset output path: {args.val_data}",
        f"Test dataset path: {args.test_data}",

    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()


