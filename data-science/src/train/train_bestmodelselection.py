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
from sklearn.ensemble import RandomForestRegressor
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
    
    # random forest specific arguments
    parser.add_argument('--n_estimators', type=int, default=500,
                        help='Number of trees in the random forest')
    parser.add_argument('--bootstrap', type=int, default=1,
                        help='Whether bootstrap samples are used when building trees')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum depth of the trees in the random forest')
    parser.add_argument('--max_features', type=str, default='auto',
                        help='Number of features to consider when looking for the best split')
    parser.add_argument('--min_samples_leaf', type=int, default=4,
                        help='Minimum number of samples required to be at a leaf node')
    parser.add_argument('--min_samples_split', type=int, default=5,
                        help='Minimum number of samples required to split an internal node')
    parser.add_argument('--random_state', type=int, default=0,
                        help='Random state for reproducibility')

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
        
        # Create models
        print("Initializing models...")
        
        # Model 1: Polynomial Regression
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=args.polynomial_degree)),
            ('linear', LinearRegression(fit_intercept=args.fit_intercept))
        ])
        
        # Model 2: Linear Regression
        linear_model = LinearRegression(fit_intercept=args.fit_intercept)
        
        # Model 3: Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            bootstrap=bool(args.bootstrap),
            max_depth=args.max_depth,
            max_features=args.max_features,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state
        )
        
        # Dictionary to store models and their metrics
        models = {
            "PolynomialRegression": poly_model,
            "LinearRegression": linear_model,
            "RandomForestRegressor": rf_model
        }
        
        model_metrics = {}
        
    except Exception as e:
        print(f"Error initializing model or loading data: {e}")
        mlflow.log_metric("data_model_init_error", 1)
        raise

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Log model hyperparameters
        if model_name == "PolynomialRegression":
            mlflow.log_param(f"{model_name}_polynomial_degree", args.polynomial_degree)
            mlflow.log_param(f"{model_name}_fit_intercept", args.fit_intercept)
        elif model_name == "LinearRegression":
            mlflow.log_param(f"{model_name}_fit_intercept", args.fit_intercept)
        elif model_name == "RandomForestRegressor":
            mlflow.log_param(f"{model_name}_n_estimators", args.n_estimators)
            mlflow.log_param(f"{model_name}_bootstrap", args.bootstrap)
            mlflow.log_param(f"{model_name}_max_depth", args.max_depth)
            mlflow.log_param(f"{model_name}_max_features", args.max_features)
            mlflow.log_param(f"{model_name}_min_samples_leaf", args.min_samples_leaf)
            mlflow.log_param(f"{model_name}_min_samples_split", args.min_samples_split)
            mlflow.log_param(f"{model_name}_random_state", args.random_state)

        # Train model
        model.fit(X_train, y_train)

        # Predict using the model
        yhat_train = model.predict(X_train)

        # Evaluate performance
        r2 = r2_score(y_train, yhat_train)
        mse = mean_squared_error(y_train, yhat_train)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_train, yhat_train)
        
        # Store metrics for comparison
        model_metrics[model_name] = {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }
        
        # Log model performance metrics
        mlflow.log_metric(f"{model_name}_train_r2", r2)
        mlflow.log_metric(f"{model_name}_train_mse", mse)
        mlflow.log_metric(f"{model_name}_train_rmse", rmse)
        mlflow.log_metric(f"{model_name}_train_mae", mae)
        
        print(f"{model_name} metrics - R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Visualize results for each model
        try:
            plt.figure(figsize=(10, 8))
            plt.scatter(y_train, yhat_train, color='black', alpha=0.6)
            plt.plot(y_train, y_train, color='blue', linewidth=3)
            plt.xlabel("Real value")
            plt.ylabel("Predicted value")
            plt.title(f"{model_name} - Predictions vs Actual Values")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{model_name}_regression_results.png", dpi=300)
            mlflow.log_artifact(f"{model_name}_regression_results.png")
            
            # Feature importance for Random Forest
            if model_name == "RandomForestRegressor":
                plt.figure(figsize=(12, 8))
                plt.title('Random Forest Feature Importance')
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                plt.bar(range(len(indices)), importances[indices], align='center')
                plt.xticks(range(len(indices)), [NUMERIC_COLS[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig("rf_feature_importance.png", dpi=300)
                mlflow.log_artifact("rf_feature_importance.png")
            
            # Feature importance for Polynomial Regression
            if model_name == "PolynomialRegression":
                poly_features = model.named_steps['poly']
                feature_names = poly_features.get_feature_names_out(NUMERIC_COLS)
                linear = model.named_steps['linear']
                coefficients = linear.coef_
                importance = np.abs(coefficients)
                indices = np.argsort(importance)[::-1][:min(15, len(importance))]
                
                plt.figure(figsize=(12, 8))
                plt.title('Polynomial Term Importance')
                plt.bar(range(len(indices)), importance[indices], align='center')
                plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig("poly_feature_importance.png", dpi=300)
                mlflow.log_artifact("poly_feature_importance.png")
                
        except Exception as e:
            print(f"Error generating plots for {model_name}: {e}")
            mlflow.log_metric(f"{model_name}_plot_generation_error", 1)

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
            
            # Send metrics for each model
            for model_name, metrics_dict in model_metrics.items():
                for metric_name, value in metrics_dict.items():
                    metric_recorder = meter.create_counter(name=f"{model_name}_{metric_name}")
                    metric_recorder.add(value)
            
            print("Successfully sent metrics to Azure Monitor")
        else:
            print("Warning: APPLICATIONINSIGHTS_CONNECTION_STRING environment variable not set. Metrics will not be sent to Azure Monitor.")
    except Exception as e:
        print(f"Error setting up Azure Monitor: {e}")
        mlflow.log_metric("monitor_setup_error", 1)
    
    # Compare models and select the best one based on R² score
    print("\nComparing models...")
    best_model_name = max(model_metrics, key=lambda x: model_metrics[x]["r2"])
    best_model = models[best_model_name]
    best_r2 = model_metrics[best_model_name]["r2"]
    
    print(f"Best model: {best_model_name} with R² score of {best_r2:.4f}")
    
    # Create comparison plot
    try:
        plt.figure(figsize=(12, 8))
        model_names = list(model_metrics.keys())
        r2_values = [model_metrics[model]["r2"] for model in model_names]
        rmse_values = [model_metrics[model]["rmse"] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width/2, r2_values, width, label='R² Score')
        ax.bar(x + width/2, rmse_values, width, label='RMSE')
        
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=300)
        mlflow.log_artifact("model_comparison.png")
        print("Model comparison plot generated successfully")
    except Exception as e:
        print(f"Error generating model comparison plot: {e}")
        mlflow.log_metric("comparison_plot_error", 1)

    # Save the best model
    try:
        print(f"Saving best model ({best_model_name}) to {args.model_output}")
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("best_model_r2", best_r2)
        mlflow.sklearn.save_model(sk_model=best_model, path=args.model_output)
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
            f"Fit intercept: {args.fit_intercept}",
            f"Random Forest n_estimators: {args.n_estimators}",
            f"Random Forest bootstrap: {args.bootstrap}",
            f"Random Forest max_depth: {args.max_depth}",
            f"Random Forest max_features: {args.max_features}",
            f"Random Forest min_samples_leaf: {args.min_samples_leaf}",
            f"Random Forest min_samples_split: {args.min_samples_split}",
            f"Random Forest random_state: {args.random_state}"
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