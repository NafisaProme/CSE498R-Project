import random
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
columns = ['Name of Commodity', 'Variation', 'Year', 'Month']

# take the user input for the above columns 
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

# take the user input for the rest of the columns, which would have integer inputs from the user
columns += ['Rainfall',	'Temperature', 'Humidity', 'Pesticide']
units = ['mm', 'Celsius', 'percentage', 'tonnes']
# options = [
#     ['Very Low (0 - 100)', 'Low (101 - 300)', 'Medium (301 - 600)', 'High (601 - 1000)', 'Very High (1000 - 1350)'],
#     ['Very Low (5 - 10)', 'Low (11 - 20)', 'Medium (21 - 30)', 'High (31 - 35)', 'Very High (36 - 45)'],
#     ['Low (51 - 60)', 'Medium (61 - 70)', 'High (71 - 80)', 'Very High (81 - 90)'],
# ]

for i in range(4, len(columns)):
    # if i == 7:
        user_input = st.number_input(f"Select {columns[i]} in {units[i - 4]}", min_value=0)
        user_inputs.append(user_input)
    # else:
    #     user_input = st.selectbox(f"Select {columns[i]} in {units[i - 4]}", options[i - 4])
    #     user_input = user_input.split('(')[1][:-1]
    #     mn, mx = int(user_input.split(' - ')[0]), int(user_input.split(' - ')[1])
    #     user_input = random.randint(mn, mx)
    #     print(user_input)
    #     user_inputs.append(user_input)

if st.button("Predict Crop Price"):
    # Make a prediction
    input_data = pd.DataFrame([user_inputs], columns=columns)
    prediction = model.predict(input_data)[0]

    # # Display the prediction
    st.write(f"The predicted crop price is: {prediction}")