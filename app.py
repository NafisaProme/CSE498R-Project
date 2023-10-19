import streamlit as st
import pandas as pd
import joblib



# Load the pre-trained model
model = joblib.load("model.pkl")

# Load the dataset
dataset_path = "dataset_2.csv"
df = pd.read_csv(dataset_path)

# Center-align inputs
st.markdown("<h1 style='text-align: center;'>Crop Price Prediction</h1>", unsafe_allow_html=True)

# Dropdown for selecting Commodity
commodity = st.selectbox("Select Commodity", df['Name of Commodity'].unique())

# Dropdown for selecting Variation
variation = st.selectbox("Select Variation", df['Variation'].unique())

# Dropdown for selecting Year
year = st.selectbox("Select Year", df['Year'].unique())

# Dropdown for selecting Month
month = st.selectbox("Select Month", df['Month'].unique())

# Dropdown for Rainfall
rainfall = st.selectbox("Select Rainfall", df['Rainfall'].unique())

# Dropdown for Temperature
temperature = st.selectbox("Select Temperature", df['Temperature'].unique())

# Dropdown for Humidity
humidity = st.selectbox("Select Humidity", df['Humidity'].unique())

# Dropdown for Pesticide
pesticide = st.selectbox("Select Pesticide", df['Pesticide'].unique())

# Combine user inputs
user_inputs = {
    'Name of Commodity': commodity,
    'Variation': variation,
    'Year': year,
    'Month': month,
    'Rainfall': rainfall,
    'Temperature': temperature,
    'Humidity': humidity,
    'Pesticide': pesticide
}

# Display the selected inputs
st.markdown("## Selected Input Parameters")
st.table(pd.DataFrame([user_inputs]))

# Predict button
if st.button("Predict Crop Price"):
    # Make a prediction
    input_data = pd.DataFrame([user_inputs])
    prediction = model.predict(input_data)[0]

    # Display the prediction
    st.header("Predicted Crop Price")
    st.write(f"The predicted crop price is: {prediction}")
