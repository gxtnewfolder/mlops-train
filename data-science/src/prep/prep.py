"""
Prepares raw data and provides training, validation, and test datasets.
"""

import argparse
from pathlib import Path
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from azureml.core import Dataset, Datastore, Workspace, Run
import os

# Initialize workspace
workspace = Run.get_context().experiment.workspace

TARGET_COL = "I_f"

NUMERIC_COLS = [
    "I_y",
    "PF",
    "e_PF",
    "d_if",
]

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data", required=True)
    parser.add_argument("--train_data", type=str, help="Path to train dataset", required=True)
    parser.add_argument("--val_data", type=str, help="Path to validation dataset", required=True)
    parser.add_argument("--test_data", type=str, help="Path to test dataset", required=True)
    return parser.parse_args()

def log_data_to_run(name, path, is_file=True):
    """Log the data path to AzureML run"""
    run = Run.get_context()
    if run.id != "offline":
        if is_file:
            # Upload individual file
            run.upload_file(name=f"{name}/{Path(path).name}", path_or_stream=path)
        else:
            # Upload folder
            run.upload_folder(name=name, path=path)
        # Log the path
        run.log(name, path)
    print(f"Logged {name}: {path}")

def main(args):
    """Read, split, save datasets, and log data paths"""
    # Read the raw data
    print("Reading raw data...")
    data = pd.read_csv(Path(args.raw_data))
    data = data[NUMERIC_COLS + [TARGET_COL]]

    # Split data
    print("Splitting data...")
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25 / 0.75, random_state=42)

    # Log dataset sizes to MLflow
    print("Logging dataset sizes...")
    mlflow.log_metric("train_size", train_data.shape[0])
    mlflow.log_metric("val_size", val_data.shape[0])
    mlflow.log_metric("test_size", test_data.shape[0])

    # Save datasets as parquet files
    print("Saving datasets...")
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.val_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_data_path = Path(args.train_data) / "train.parquet"
    val_data_path = Path(args.val_data) / "val.parquet"
    test_data_path = Path(args.test_data) / "test.parquet"

    train_data.to_parquet(train_data_path)
    val_data.to_parquet(val_data_path)
    test_data.to_parquet(test_data_path)

    # Log paths to AzureML Run
    print("Logging data paths to AzureML...")
    log_data_to_run("train_data_path", str(train_data_path))
    log_data_to_run("val_data_path", str(val_data_path))
    log_data_to_run("test_data_path", str(test_data_path))

    print("Data preparation completed successfully.")

if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Validation dataset output path: {args.val_data}")
    print(f"Test dataset output path: {args.test_data}")

    main(args)

    mlflow.end_run()