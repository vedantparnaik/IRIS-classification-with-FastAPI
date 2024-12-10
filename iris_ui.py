import streamlit as st
import requests

# Streamlit UI
st.title("Iris Species Prediction")
st.write("Enter the details of the iris flower to predict its species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# Button to trigger prediction
if st.button("Predict"):
    # Make the API request
    url = "http://127.0.0.1:8000/predict"  # Ensure FastAPI is running on this address
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Predicted Species: {prediction['predicted_species_name']}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to connect to the API. Error: {str(e)}")
