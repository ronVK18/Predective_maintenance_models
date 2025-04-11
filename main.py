from fastapi import FastAPI
from pydantic import BaseModel,Field
from joblib import load
import numpy as np
import pandas as pd
from typing import Optional

app = FastAPI()

# Load your model once when the app starts
model = load('RF.pkl')

# Define input data model
class PumpData(BaseModel):
    Type: str  # or int if you expect numerical type
    Air_temperature_K: float = Field(..., alias="Air temperature [K]")
    Process_temperature_K: float = Field(..., alias="Process temperature [K]")
    Rotational_speed_rpm: int = Field(..., alias="Rotational speed [rpm]")
    Torque_Nm: float = Field(..., alias="Torque [Nm]")
    Tool_wear_min: int = Field(..., alias="Tool wear [min]")

def format_input(input_data: dict):
    """
    Processes input data into model-ready format
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    # Feature Engineering
    input_df['temperature_difference'] = input_df['Process temperature [K]'] - input_df['Air temperature [K]']
    input_df['Mechanical Power [W]'] = np.round(
        (input_df['Torque [Nm]'] * input_df['Rotational speed [rpm]'] * 2 * np.pi) / 60, 4)

    # Ensure columns are in the same order as training
    final_cols = [
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'temperature_difference', 'Mechanical Power [W]'
    ]
    
    return input_df[final_cols]

@app.post("/predict")
async def predict_failure(pump_data: PumpData):
    """
    Endpoint to predict machine failure probability
    """
    # Convert Pydantic model to dict and handle field aliases
    input_dict = pump_data.dict(by_alias=True)
    
    # Format input for model
    model_input = format_input(input_dict)
    
    # Make prediction
    prediction = model.predict(model_input)
    
    # If your model outputs probabilities, you might want to return them
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(model_input)[0][1]  # Assuming binary classification
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "message": "High risk of failure" if prediction[0] == 1 else "Low risk of failure"
        }
    else:
        return {"prediction": int(prediction[0])}

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "OK", "message": "Pump Failure Prediction Service is running"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)