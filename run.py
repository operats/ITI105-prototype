import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta
import joblib
import base64
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the model and encoders
model = joblib.load("LogisticRegression_small.joblib")
#one_hot_encoded_cat = joblib.load("one_hot_encoded_cat.joblib")

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Single', 'Batch'])
if app_mode == 'Home':
    st.title('NYP ITI105 Fraud Prediction App')
    #st.image('hipster_loan-1.jpg')
    st.write('App created by Team 7')
    st.write('Select at the sidebar:')
    st.write('>>> Single: Enter the inputs yourself and make a prediction.')
    st.write('>>> Batch:  Upload a csv file and make multiple predictions.')

elif app_mode == 'Batch':

    # Create a Streamlit app
    st.title("Fraud Prediction by batch upload")
    st.write('Only csv files are supported')
    
    uploaded_file = st.file_uploader("Choose a csv file")
    if uploaded_file is not None: 
        batch_df = pd.read_csv(uploaded_file)
        st.write(batch_df)

elif app_mode == 'Single':

    # Create a Streamlit app
    st.title("Interactive Single Fraud Prediction")
    
    # Add input fields for each column
    #index = st.text_input("Unique Identifier")
    trans_date = st.date_input("Transaction Date", value=datetime.date(2019, 1, 2))
    trans_time = st.time_input("Transaction Time", value=datetime.time(1,6,37))
    #trans_hour = st.numer_input("Transaction Hour", value=1)
    cc_num = st.text_input("Credit Card Number", value="4613314721966")
    merchant = st.text_input("Merchant Name", value="fraud_Rutherford-Mertz")
    #category = st.selectbox("Category of Merchant", ["grocery_pos", "dining_pos", ...], value="grocery_pos")
    category = st.text_input("Category of Merchant", value="grocery_pos")
    amt = st.number_input("Amount of Transaction", value=281.06)
    first = st.text_input("First Name", value="Jason")
    last = st.text_input("Last Name", value="Murphy")
    gender = st.selectbox("Gender", ["M", "F"], index=0)
    street = st.text_input("Street Address", value="542 Steve Curve Suite 011")
    city = st.text_input("City", value="Collettsville")
    state = st.text_input("State", value="NC")
    zip = st.text_input("Zip", value="28611")
    lat = st.number_input("Latitude", value=35.9946)
    long = st.number_input("Longitude", value=-81.7266)
    city_pop = st.number_input("City Population", value=885)
    job = st.text_input("Job", value="Soil scientist")
    dob = st.date_input("Date of Birth", value=datetime.date(1988, 9, 15))
    trans_num = st.text_input("Transaction Number", value="e8a81877ae9a0a7f883e15cb39dc4022")
    unix_time = st.number_input("UNIX Time", value=1325466397)
    merch_lat = st.number_input("Merchant Latitude", value=36.430124)
    merch_long = st.number_input("Merchant Longitude", value=-81.17948299999999)
    #days = st.selectbox("Days", ["Monday", "Tuesday", ...])
    
    # Define the categorical columns
    cat_cols = ["gender", "state", "category", "job", "days"]
    
    # Add a button
    if st.button("Predict"):
        # Create a dataframe with the input values
        input_df = pd.DataFrame({
    #        "index": [index],
            "trans_date": [trans_date],
            "trans_time": [trans_time],
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
            "merch_long": [merch_long]
    #        "days": [days]
        })
    
        # Process trans_date_trans_time
        input_df['date'] = pd.to_datetime(input_df['trans_date'])
        input_df['days'] = input_df['date'].dt.day_name()
        #input_df['trans_time'] = pd.to_datetime(input_df['trans_time']).dt.hour
        #input_df['hour'] = pd.to_datetime(input_df['trans_time']).dt.hour
        #input_df['hour'] = input_df['trans_time'].dt.hour
        #input_df['hour'] = input_df['trans_time'].apply(lambda x: x.hour)
        input_df['hour'] = input_df['trans_time'][0].hour
    
        # Delete the original category columns
        input_df.drop(columns=['trans_date','trans_time','date'], inplace=True)
        #input_df.drop(columns=['trans_date','trans_time','date','time'], inplace=True)
        #input_df.drop(columns=['gender', 'state', 'category', 'job', 'days'], inplace=True)
        input_df.drop(columns=['cc_num', 'city', 'dob', 'first', 'last', 'lat', 'long', 'merch_lat', 'merch_long', 'merchant', 'street', 'trans_num', 'unix_time', 'zip'], inplace=True)
    
        # Add the OHE categories
        input_df['state_AR'] = [0,]
        input_df['state_AZ'] = [0,]
        input_df['state_CA'] = [0,]
        input_df['state_HI'] = [0,]
        input_df['state_IA'] = [0,]
        input_df['state_IL'] = [0,]
        input_df['state_IN'] = [0,]
        input_df['state_KY'] = [0,]
        input_df['state_LA'] = [0,]
        input_df['state_MA'] = [0,]
        input_df['state_MI'] = [0,]
        input_df['state_MO'] = [0,]
        input_df['state_NC'] = [0,]
        input_df['state_NY'] = [0,]
        input_df['state_OH'] = [0,]
        input_df['state_OK'] = [0,]
        input_df['state_OR'] = [0,]
        input_df['state_PA'] = [0,]
        input_df['state_SC'] = [0,]
        input_df['state_TX'] = [0,]
        input_df['state_WA'] = [0,]
        input_df['state_WI'] = [0,]
        input_df['category_grocery_net'] = [0,]
        input_df['category_grocery_pos'] = [0,]
        input_df['category_misc_net'] = [0,]
        input_df['category_misc_pos'] = [0,]
        input_df['category_shopping_net'] = [0,]
        input_df['category_shopping_pos'] = [0,]
        input_df['job_Aeronautical engineer'] = [0,]
        input_df['job_Agricultural consultant'] = [0,]
        input_df['job_Arts development officer'] = [0,]
        input_df['job_Barista'] = [0,]
        input_df['job_Barrister'] = [0,]
        input_df['job_Camera operator'] = [0,]
        input_df['job_Chartered loss adjuster'] = [0,]
        input_df['job_Corporate investment banker'] = [0,]
        input_df['job_Editor, commissioning'] = [0,]
        input_df['job_Emergency planning/management officer'] = [0,]
        input_df['job_Engineer, biomedical'] = [0,]
        input_df['job_Equities trader'] = [0,]
        input_df['job_Film/video editor'] = [0,]
        input_df['job_Financial adviser'] = [0,]
        input_df['job_Firefighter'] = [0,]
        input_df['job_Horticultural consultant'] = [0,]
        input_df['job_IT trainer'] = [0,]
        input_df['job_Insurance risk surveyor'] = [0,]
        input_df['job_Investment banker, corporate'] = [0,]
        input_df['job_Land/geomatics surveyor'] = [0,]
        input_df['job_Magazine features editor'] = [0,]
        input_df['job_Osteopath'] = [0,]
        input_df['job_Petroleum engineer'] = [0,]
        input_df['job_Physiotherapist'] = [0,]
        input_df['job_Production assistant, radio'] = [0,]
        input_df['job_Quantity surveyor'] = [0,]
        input_df['job_Research scientist (physical sciences)'] = [0,]
        input_df['job_Sales professional, IT'] = [0,]
        input_df['job_Scientist, audiological'] = [0,]
        input_df['job_Seismic interpreter'] = [0,]
        input_df['job_Soil scientist'] = [0,]
        input_df['job_Special educational needs teacher'] = [0,]
        input_df['job_Stage manager'] = [0,]
        input_df['job_Surveyor, rural practice'] = [0,]
        input_df['job_Tax adviser'] = [0,]
        input_df['job_Teacher, secondary school'] = [0,]
        
        
        # One-hot encode the categorical columns
        #input_df = one_hot_encoded_cat.transform(input_df[["gender", "state", "category", "job", "days"]])
        input_df = pd.get_dummies(input_df, columns=['gender', 'state', 'category', 'job', 'days'])
        # Call the one_hot_encoded_cat function with the input data and categorical columns
        #input_df = one_hot_encoded_cat_func(input_df, cat_cols)
    
        # Remove extra unused columns
        input_df.drop(columns=['days_Wednesday'], inplace=True)
        
        # Make a prediction using the model
        prediction = model.predict(input_df)
    
        # Display the prediction
        st.write("Prediction:", prediction)
    #    if prediction[0] == 1:
    #        st.error('According to the model, this is a fraud!')
            #st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">', unsafe_allow_html=True)
    #    elif prediction[0] == 0:
    #        st.success('Congratulations! This transaction is legitimate.')
            #st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">', unsafe_allow_html=True)
