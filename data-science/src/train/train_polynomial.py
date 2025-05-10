"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from azure.monitor.opentelemetry.exporter import AzureMonitorMetricExporter

TARGET_COL = "I_f"

NUMERIC_COLS = [
    "I_y",
    "PF",
    "e_PF",
    "d_if",
]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # polynomial regression specific arguments
    parser.add_argument('--polynomial_degree', type=int, default=3,
                        help='Degree of the polynomial features')
    parser.add_argument('--fit_intercept', type=bool, default=True,
                        help='Whether to fit the intercept for the linear regression')

    args = parser.parse_args()

    return args

# def save_dataset_as_asset(train_data, asset_name, asset_description, datastore_name="workspaceblobstore"):
#     """
#     Save the dataset as a registered data asset in Azure ML.
#     """
#     # Initialize Azure ML Client
#     ml_client = MLClient(DefaultAzureCredential(), subscription_id="65277b9d-88e1-4586-9111-b95e38a4426c", resource_group_name="tl01-ai-ml", workspace_name="aml-tl01-ai-ml")

#     train_data.to_csv("train_data.csv")
    
#     current_time = datetime.datetime.now()
#     partition_path = current_time.strftime("%Y/%m/%d/%H")
    
#     # Register the dataset as a data asset
#     data_asset = Data(
#         name=asset_name,
#         description=asset_description,
#         type="uri_file",
#         path=f"azureml://datastores/{datastore_name}/paths/modelDataCollector/sync_train_data/{partition_path}/{asset_name}.csv",
#     )
#     ml_client.data.create_or_update(data_asset)
#     print(f"Dataset registered in Azure ML with time-based partition: {partition_path}")

def main(args):
    '''Read train dataset, train model, save trained model'''
    try:
        # Read train data
        print(f"Loading training data from {args.train_data}")
        train_data = pd.read_parquet(Path(args.train_data))
        print(f"Training data loaded successfully with {len(train_data)} records")
        
        # Check for required columns
        missing_cols = [col for col in NUMERIC_COLS + [TARGET_COL] if col not in train_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in training data: {missing_cols}")

        # Split the data into input(X) and output(y)
        y_train = train_data[TARGET_COL]
        X_train = train_data[NUMERIC_COLS]

        # Log feature importance analysis before training
        print(f"Feature columns: {NUMERIC_COLS}")
        print(f"Target column: {TARGET_COL}")
        
        # Create a polynomial regression model with degree 3
        print("Initializing Polynomial Regression model with degree 3...")
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=args.polynomial_degree)),
            ('linear', LinearRegression(fit_intercept=args.fit_intercept))
        ])
    except Exception as e:
        print(f"Error initializing model or loading data: {e}")
        mlflow.log_metric("data_model_init_error", 1)
        raise

    # log model hyperparameters
    mlflow.log_param("model", "PolynomialRegression")
    mlflow.log_param("polynomial_degree", args.polynomial_degree)
    mlflow.log_param("fit_intercept", args.fit_intercept)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Setup Azure Monitor with secure credentials handling
    try:
        # Get connection string from environment variable
        connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if connection_string:
            exporter = AzureMonitorMetricExporter.from_connection_string(connection_string)
            reader = PeriodicExportingMetricReader(exporter)
            provider = MeterProvider(metric_readers=[reader])
            metrics.set_meter_provider(provider)
            meter = metrics.get_meter(__name__)
            r2_value_recorder = meter.create_counter(name="r2_value")
            mse_value_recorder = meter.create_counter(name="mse_value")
            r2_value_recorder.add(r2)
            mse_value_recorder.add(mse)
            print("Successfully sent metrics to Azure Monitor")
        else:
            print("Warning: APPLICATIONINSIGHTS_CONNECTION_STRING environment variable not set. Metrics will not be sent to Azure Monitor.")
    except Exception as e:
        print(f"Error setting up Azure Monitor: {e}")
        mlflow.log_metric("monitor_setup_error", 1)
    

    # Visualize results
    try:
        print("Generating regression plots...")
        plt.figure(figsize=(10, 8))
        plt.scatter(y_train, yhat_train, color='black', alpha=0.6)
        plt.plot(y_train, y_train, color='blue', linewidth=3)
        plt.xlabel("Real value")
        plt.ylabel("Predicted value")
        plt.title("Model Predictions vs Actual Values")
        plt.grid(True, alpha=0.3)
        plt.savefig("regression_results.png", dpi=300)
        mlflow.log_artifact("regression_results.png")
        
        # Get polynomial feature names
        poly_features = model.named_steps['poly']
        feature_names = poly_features.get_feature_names_out(NUMERIC_COLS)
        
        # Get coefficients from the linear regression model
        linear_model = model.named_steps['linear']
        coefficients = linear_model.coef_
        
        # Plot the top 15 most important features by coefficient magnitude
        importance = np.abs(coefficients)
        indices = np.argsort(importance)[::-1][:min(15, len(importance))]
        
        plt.figure(figsize=(12, 8))
        plt.title('Polynomial Term Importance')
        plt.bar(range(len(indices)), importance[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300)
        mlflow.log_artifact("feature_importance.png")
        print("Plots generated and saved successfully")
    except Exception as e:
        print(f"Error generating plots: {e}")
        mlflow.log_metric("plot_generation_error", 1)

    # Save the model
    try:
        print(f"Saving model to {args.model_output}")
        mlflow.sklearn.save_model(sk_model=model, path=args.model_output)
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")
        mlflow.log_metric("model_save_error", 1)
        raise


if __name__ == "__main__":
    try:
        print("Starting model training process")
        mlflow.start_run()

        # ---------- Parse Arguments ----------- #
        # -------------------------------------- #
        args = parse_args()

        # Log all parameters for reproducibility
        print("Training configuration:")
        lines = [
            f"Train dataset input path: {args.train_data}",
            f"Model output path: {args.model_output}",
            f"Polynomial degree: {args.polynomial_degree}",
            f"Fit intercept: {args.fit_intercept}"
        ]

        for line in lines:
            print(line)

        # Execute main training flow
        main(args)
        print("Training completed successfully")

    except Exception as e:
        print(f"ERROR: Training process failed: {e}")
        mlflow.log_metric("training_failed", 1)
        # Still end the run even if we had an error
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
    finally:
        # Ensure run is always ended properly
        if mlflow.active_run():
            mlflow.end_run()