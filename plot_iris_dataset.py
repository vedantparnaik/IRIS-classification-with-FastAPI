import pickle
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train and save the model
def train_and_save_model(model_path="iris_model.pkl"):
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the trained model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

# Load the ML model
def load_model(model_path="iris_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Predict using the loaded model
def predict_species(features, model):
    # Convert input features to a 2D array
    input_data = np.array([features])
    predicted_class = model.predict(input_data)[0]
    return predicted_class

# Business logic for handling feature extraction and calling prediction
def handle_prediction_request(features, model):
    # Ensure features are in the correct format (list of feature values)
    return predict_species(features, model)

# Main logic
if __name__ == "__main__":
    # Train and save the model
    train_and_save_model()

    # Load the model
    model = load_model()

    # Example: Predict species for a new sample
    example_features = [5.1, 3.5, 1.4, 0.2]  # Sepal length, width, petal length, width
    predicted_class = handle_prediction_request(example_features, model)
    print(f"Predicted class: {predicted_class}")
