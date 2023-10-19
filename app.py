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
commodity = st.selectbox("Select Commodity", df['Name of Commodity'])

# Dropdown for selecting Variation
variation = st.selectbox("Select Variation", df['Variation'])

# Dropdown for selecting Year
year = st.selectbox("Select Year", df['Year'])

# Dropdown for selecting Month
month = st.selectbox("Select Month", df['Month'])

# Dropdown for Rainfall
rainfall = st.selectbox("Select Rainfall", df['Rainfall'])

# Dropdown for Temperature
temperature = st.selectbox("Select Temperature", df['Temperature'])

# Dropdown for Humidity
humidity = st.selectbox("Select Humidity", df['Humidity'])

# Dropdown for Pesticide
pesticide = st.selectbox("Select Pesticide", df['Pesticide'])

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

# user_inputs = [0, 0, 0, 0, 27, 27, 83, 4333]

# Display the selected inputs
st.markdown("## Selected Input Parameters")
st.table(pd.DataFrame([user_inputs]))

# Predict button
if st.button("Predict Crop Price"):
    # Make a prediction
    input_data = pd.DataFrame([user_inputs])
    prediction = model.predict(input_data)[0]

    # # Display the prediction
    st.header("Predicted Crop Price")
    st.write(f"The predicted crop price is: {prediction}")

# Temperature dropdown options
# Very Low(20-25)
# Low(26-30)
# Medium(31-35)
# High(36-40)
# Very High(41-45)