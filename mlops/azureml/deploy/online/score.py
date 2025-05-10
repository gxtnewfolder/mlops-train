import os
import pandas as pd
import numpy as np
import json
import logging
import traceback
import time
from azureml.ai.monitoring import Collector
from azureml.core.model import Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
global model, inputs_collector, outputs_collector, inputs_outputs_collector, model_name

def init():
    """Initialize the model and data collectors.
    This function is called when the container is initialized.
    """
    global model, inputs_collector, outputs_collector, inputs_outputs_collector, model_name
    
    start_time = time.time()
    try:
        logger.info("Starting model initialization...")
        
        # Get model path
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
        model_name = "synchronous-machines-model"
        
        # Load model (this would be implementation-specific based on your model type)
        # Example for scikit-learn model:
        import joblib
        model = joblib.load(os.path.join(model_path, "model.pkl"))
        
        # Instantiate collectors with appropriate names, align with deployment spec
        inputs_collector = Collector(name="model_inputs")
        outputs_collector = Collector(name="model_outputs")
        inputs_outputs_collector = Collector(name="model_inputs_outputs")
        
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        logger.error(traceback.format_exc())
        # Re-raise the exception to signal initialization failure
        raise

def run(data):
    """Run the model on the input data.
    
    Args:
        data (str): JSON string containing the input data.
        
    Returns:
        dict: Model predictions as a dictionary.
    """
    logger.info("Received prediction request")
    start_time = time.time()
    
    try:
        # Parse and preprocess the input data
        json_data = json.loads(data)
        pdf_data = preprocess(json_data)
        
        # Convert to DataFrame
        input_df = pd.DataFrame(pdf_data)
        logger.info(f"Input data shape: {input_df.shape}")
        
        # Validate input data
        if input_df.empty:
            raise ValueError("Empty input data received")
        
        # Collect inputs data, store correlation_context
        context = inputs_collector.collect(input_df)
        
        # Perform prediction
        output_df = predict(input_df)
        
        # Collect outputs data with correlation context
        outputs_collector.collect(output_df, context)
        
        # Join inputs and outputs for drift detection
        input_output_df = input_df.join(output_df)
        
        # Collect inputs and outputs together
        inputs_outputs_collector.collect(input_output_df, context)
        
        logger.info(f"Prediction completed in {time.time() - start_time:.3f} seconds")
        return output_df.to_dict()
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "success": False}

def preprocess(json_data):
    """Preprocess the input JSON data.
    
    Args:
        json_data (dict): Input data as a dictionary.
        
    Returns:
        dict: Preprocessed data ready for model input.
    """
    try:
        # Extract data from json payload
        if "data" not in json_data:
            raise ValueError("Input JSON must contain a 'data' key")
            
        data = json_data["data"]
        
        # Perform any necessary data transformations here
        # Example: data normalization, feature engineering, etc.
        
        return data
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def predict(input_df):
    """Make predictions using the loaded model.
    
    Args:
        input_df (pd.DataFrame): Input data as a pandas DataFrame.
        
    Returns:
        pd.DataFrame: Predictions as a pandas DataFrame.
    """
    try:
        # Make predictions with the model
        # The actual implementation depends on your model type
        predictions = model.predict(input_df)
        
        # Create DataFrame with predictions
        output_df = pd.DataFrame({
            "prediction": predictions,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Add prediction probabilities if applicable
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(input_df)
            if probas.shape[1] == 2:  # Binary classification
                output_df['probability'] = probas[:, 1]
        
        return output_df
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise
