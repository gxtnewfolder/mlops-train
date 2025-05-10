"""
Trains ML model using hyperparameter tuning with Azure ML's hyperdrive.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

from azure.ai.ml import MLClient
from azure.ai.ml.entities import HyperDriveConfig, RandomParameterSampling
from azure.ai.ml.sweep import Choice, Uniform, LogUniform
from azure.identity import DefaultAzureCredential

TARGET_COL = "I_f"

NUMERIC_COLS = [
    "I_y",
    "PF",
    "e_PF",
    "d_if",
]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train_hyperdrive")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, help="Azure resource group name")
    parser.add_argument("--workspace_name", type=str, help="Azure ML workspace name")

    args = parser.parse_args()

    return args

def main(args):
    '''Configure and run hyperparameter tuning'''
    try:
        # Initialize MLClient
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name
        )

        # Define the hyperparameter search space
        search_space = {
            "n_estimators": Choice([100, 200, 300, 400, 500]),
            "max_depth": Choice([5, 10, 15, 20, None]),
            "min_samples_leaf": Choice([1, 2, 4, 6, 8]),
            "min_samples_split": Choice([2, 4, 6, 8, 10]),
            "max_features": Choice(['auto', 'sqrt', 'log2']),
            "bootstrap": Choice([True, False])
        }

        # Define the sampling method
        sampling = RandomParameterSampling(search_space)

        # Define the primary metric
        primary_metric = "train_r2"
        goal = "Maximize"

        # Create the hyperdrive configuration
        hyperdrive_config = HyperDriveConfig(
            sampling=sampling,
            primary_metric_name=primary_metric,
            primary_metric_goal=goal,
            max_total_trials=20,
            max_concurrent_trials=4
        )

        # Create the training job
        job = ml_client.jobs.create_or_update(
            name="synchronous-machines-hyperdrive",
            type="command",
            inputs={
                "train_data": args.train_data
            },
            outputs={
                "model_output": args.model_output
            },
            code="../../../data-science/src/train",
            command="python train.py --train_data ${{inputs.train_data}} --model_output ${{outputs.model_output}} --n_estimators ${{search_space.n_estimators}} --max_depth ${{search_space.max_depth}} --min_samples_leaf ${{search_space.min_samples_leaf}} --min_samples_split ${{search_space.min_samples_split}} --max_features ${{search_space.max_features}} --bootstrap ${{search_space.bootstrap}}",
            environment="azureml:synchronous-machines-env@latest",
            compute="azureml:ado-instance-tl01",
            hyperdrive_config=hyperdrive_config
        )

        # Submit the job
        ml_client.jobs.begin_create_or_update(job)
        print("Hyperparameter tuning job submitted successfully")

    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
        raise

if __name__ == "__main__":
    try:
        print("Starting hyperparameter tuning process")
        args = parse_args()
        main(args)
        print("Hyperparameter tuning completed successfully")

    except Exception as e:
        print(f"ERROR: Hyperparameter tuning failed: {e}")
        raise 