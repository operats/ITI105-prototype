import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the model and encoders
model = joblib.load("LogisticRegression.joblib")
one_hot_encoded_cat = joblib.load("one_hot_encoded_cat.joblib")

# Create a Streamlit app
st.title("Fraud Prediction App")

# Add input fields for each column
index = st.text_input("Unique Identifier")
trans_date_trans_time = st.date_input("Transaction DateTime")
cc_num = st.text_input("Credit Card Number")
merchant = st.text_input("Merchant Name")
category = st.selectbox("Category of Merchant", ["Category 1", "Category 2", ...])
amt = st.number_input("Amount of Transaction")
first = st.text_input("First Name")
last = st.text_input("Last Name")
gender = st.selectbox("Gender", ["Male", "Female"])
street = st.text_input("Street Address")
city = st.text_input("City")
state = st.text_input("State")
zip = st.text_input("Zip")
lat = st.number_input("Latitude")
long = st.number_input("Longitude")
city_pop = st.number_input("City Population")
job = st.text_input("Job")
dob = st.date_input("Date of Birth")
trans_num = st.text_input("Transaction Number")
unix_time = st.number_input("UNIX Time")
merch_lat = st.number_input("Merchant Latitude")
merch_long = st.number_input("Merchant Longitude")
days = st.selectbox("Days", ["Monday", "Tuesday", ...])

# Define the categorical columns
cat_cols = ["gender", "state", "category", "job", "days"]

# Add a button
if st.button("Predict"):
    # Create a dataframe with the input values
    input_df = pd.DataFrame({
        "index": [index],
        "trans_date_trans_time": [trans_date_trans_time],
        "cc_num": [cc_num],
        "merchant": [merchant],
        "category": [category],
        "amt": [amt],
        "first": [first],
        "last": [last],
        "gender": [gender],
        "street": [street],
        "city": [city],
        "state": [state],
        "zip": [zip],
        "lat": [lat],
        "long": [long],
        "city_pop": [city_pop],
        "job": [job],
        "dob": [dob],
        "trans_num": [trans_num],
        "unix_time": [unix_time],
        "merch_lat": [merch_lat],
        "merch_long": [merch_long],
        "days": [days]
    })

    # One-hot encode the categorical columns
    #input_df = one_hot_encoded_cat.transform(input_df[["gender", "state", "category", "job", "days"]])
    input_df = pd.get_dummies(input_df, columns=['gender', 'state', 'category', 'job', 'days'])
    # Call the one_hot_encoded_cat function with the input data and categorical columns
    #input_df = one_hot_encoded_cat_func(input_df, cat_cols)

    # Make a prediction using the model
    prediction = model.predict(input_df)

    # Display the prediction
    st.write("Prediction:", prediction)
    if prediction[0] == 1:
        st.error('According to the model, this is a fraud!')
        #st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">', unsafe_allow_html=True)
    elif prediction[0] == 0:
        st.success('Congratulations! This transaction is legitimate.')
        #st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">', unsafe_allow_html=True)
