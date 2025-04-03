import os
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Get the absolute path to the model file
root_folder = os.path.dirname(__file__)
model_path = os.path.join(root_folder, 'my_model.keras')

# Load the model
model = load_model(model_path, compile=False)

# load scaler label and one hot encoder are saved in pickle file
scaler_path = os.path.join(root_folder, 'scaler.pkl')
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

label_encode_path = os.path.join(root_folder, 'label_encoder_gender.pkl')
with open(label_encode_path, 'rb') as file:
    label_encoder_gender = pickle.load(file)

one_hot_encode_path = os.path.join(root_folder, 'ohe_geography.pkl')
with open(one_hot_encode_path, 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

## Streamlit app
st.title('Customer Churn Prediction')

# User input fields
# one_hot_encoder_geo.categories_[0] contains ['France', 'Germany', 'Spain'],
# the dropdown will allow the user to select one of these options.
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0]) # one_hot_encoder_geo.categories_[0] is the list of unique values
gender = st.selectbox('Gender', label_encoder_gender.classes_) # classes_ is the list of unique values
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data in dictionary format
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
})

st.write("Input Data:", input_data)


# One-hot encode 'Geography'
geo_encoded = one_hot_encoder_geo.transform(input_data[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Drop the original 'Geography' column as it is now one-hot encoded
input_data = input_data.drop(columns=['Geography'])

# Scale the input data
input_data_scaled = scaler.transform(input_data)
st.write("Scaled Input Data:", input_data_scaled)
# Predict churn
prediction = model.predict(input_data_scaled)

# This generates predictions for the input data.
# For binary classification, the model outputs a 2D array where each row corresponds to a sample, 
# and the single value in each row is the probability of the positive class (e.g., "churn").
# prediction = [[0.78]] - the probability of the customer churning
prediction_proba = prediction[0][0]

st.write("Raw Prediction Output:", prediction)
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')