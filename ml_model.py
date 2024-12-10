import pickle
import numpy as np

# Load the ML model
def load_model(model_path="iris_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Initialize the model
model = load_model()

# Predict house price using the loaded model
def predict_iris(features):
    # Convert input features to a 2D array
    input_data = np.array([features])
    predicted_species = model.predict(input_data)[0]
    return int(predicted_species)


# Business logic for handling feature extraction and calling prediction
def handle_prediction_request(features):
    # Convert input data into the required list format
    feature_list = [
        features.sepal_length, features.sepal_width,
        features.petal_length, features.petal_width
    ]
    return predict_iris(feature_list)
