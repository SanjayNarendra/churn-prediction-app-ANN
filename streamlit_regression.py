import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit App Layout
st.set_page_config(page_title="Salary Prediction", layout="centered")
st.title('Estimated Salary Prediction')

# User input
st.header("Customer Details")
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 65)
balance = st.number_input('💰 Account Balance (€)', min_value=0.0,)
credit_score = st.number_input('Credit Score',  300, 900)
exited = st.selectbox('Excited bank', ['Yes', 'No'])
tenure = st.slider('Tenure (Years)', 0, 10)
num_of_products = st.slider('Number of Products Used', 1, 4)
has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])

has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0
exited = 1 if exited == 'Yes' else 0

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]  
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# concatenate one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

## Prediction Button
if st.button('Analyze Customer'):
    prediction = model.predict(input_data_scaled)
    prediction_salary = prediction[0][0]
    
    st.metric(label="Predicted Salary: ", value=f'€ {prediction_salary:.2f}')

