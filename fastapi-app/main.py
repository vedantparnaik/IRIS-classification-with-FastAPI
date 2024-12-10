from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml_model import handle_prediction_request

# Initialize FastAPI
app = FastAPI()

# Input schema for Iris prediction
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"Message": "Welcome to the Iris Species Prediction API!"}

@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        # Delegate to the combined business/ML logic
        predicted_species_index = handle_prediction_request(features)
        # Map species index to name
        iris_target_names = ["setosa", "versicolor", "virginica"]
        predicted_species_name = iris_target_names[predicted_species_index]
        return {"predicted_species_index": predicted_species_index, "predicted_species_name": predicted_species_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "API is running and healthy!"}

