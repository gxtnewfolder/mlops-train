"""
Registers trained ML model if deploy flag is True.
"""

import argparse
from pathlib import Path
import pickle
import mlflow

import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of eval results')
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info JSON"
    )
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args


def main(args):
    '''Loads model, registers it if deploy flag is True'''
    try:
        # Read deployment flag from evaluation output
        deploy_flag_path = Path(args.evaluation_output) / "deploy_flag"
        if not deploy_flag_path.exists():
            print(f"Warning: Deploy flag file not found at {deploy_flag_path}")
            deploy_flag = 0
        else:
            with open(deploy_flag_path, 'rb') as infile:
                deploy_flag = int(infile.read())
            
        # Log the original deployment decision
        print(f"Evaluation deploy flag: {deploy_flag}")
        mlflow.log_metric("evaluation_deploy_flag", int(deploy_flag))
        
        # Use the evaluation result for deployment decision
        if deploy_flag == 1:

            print(f"Registering model: {args.model_name}")

            try:
                # load model
                print(f"Loading model from {args.model_path}")
                model = mlflow.sklearn.load_model(args.model_path)
                
                # log model using mlflow
                print("Logging model to MLflow")
                mlflow.sklearn.log_model(model, args.model_name)

                # register logged model using mlflow
                run_id = mlflow.active_run().info.run_id
                model_uri = f'runs:/{run_id}/{args.model_name}'
                print(f"Registering model with URI: {model_uri}")
                mlflow_model = mlflow.register_model(model_uri, args.model_name)
                model_version = mlflow_model.version
                print(f"Model registered successfully as {args.model_name} version {model_version}")

                # write model info
                model_info = {"id": "{0}:{1}".format(args.model_name, model_version)}
                output_path = os.path.join(args.model_info_output_path, "model_info.json")
                
                print(f"Writing model info to {output_path}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
                with open(output_path, "w") as of:
                    json.dump(model_info, fp=of, indent=2)
                print("Model info written successfully")
                
                # Log additional metadata
                mlflow.log_metric("model_registered", 1)
                mlflow.log_metric("model_version", float(model_version))
                
            except Exception as e:
                print(f"Error during model registration: {e}")
                mlflow.log_metric("registration_error", 1)
                raise

        else:
            print("Model WILL NOT be registered based on evaluation results")
            mlflow.log_metric("model_registered", 0)
            
    except Exception as e:
        print(f"Error in model registration process: {e}")
        mlflow.log_metric("registration_process_error", 1)
        raise

if __name__ == "__main__":
    try:
        print("Starting model registration process")
        mlflow.start_run()
        
        # ---------- Parse Arguments ----------- #
        # -------------------------------------- #
        args = parse_args()
        
        lines = [
            f"Model name: {args.model_name}",
            f"Model path: {args.model_path}",
            f"Evaluation output path: {args.evaluation_output}",
            f"Model info output path: {args.model_info_output_path}",
        ]

        for line in lines:
            print(line)

        main(args)
        print("Model registration process completed successfully")
        
    except Exception as e:
        print(f"ERROR: Model registration failed: {e}")
        mlflow.log_metric("registration_failed", 1)
        # Still end the run even if we had an error
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
    finally:
        # Ensure run is always ended properly
        if mlflow.active_run():
            mlflow.end_run()
