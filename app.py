import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load("model.pkl")

# Loading the raw dataset
dataset_path = "dataset_2.csv" 
df = pd.read_csv(dataset_path)
df = df.drop(columns=['Price'])

# Create a dictionary to store the selected values for each column
user_inputs = []

columns = df.columns
for column in columns:
    # Get unique values for the current column
    unique_values = df[column].unique()

    # Create a Streamlit dropdown for selecting unique values
    selected_value = st.selectbox(f"Select a value for {column}", unique_values)

    ind = 0
    for i in range(0, len(unique_values)):
        if selected_value == unique_values[i]:
            ind = i
            break

    # Store the selected value in the user_input
    user_inputs.append(ind)

if st.button("Predict Crop Price"):
    # Make a prediction
    input_data = pd.DataFrame([user_inputs])
    prediction = model.predict(input_data)[0]

    # # Display the prediction
    st.header("Predicted Crop Price")
    st.write(f"The predicted crop price is: {prediction}")