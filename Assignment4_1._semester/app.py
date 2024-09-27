import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
from streamlit_shap import st_shap

# Page configuration
st.set_page_config(
    page_title="Starting a business?",
    page_icon="👨‍💼")

st.title('Predict likelighood of starting a business')

# Display an image
st.image('https://kinore.com/wp-content/uploads/2019/03/Ideas.jpg', caption='Business', use_column_width=True)

# Load the dataset and some manipulation
data = pd.read_parquet('GEM 2020 APS Global Individual Level Data_Jan2022.parquet') 
df_1 = data[['gemhhinc', 'gemeduc', 'nbmedial', 'age', 'bstart', 'region']]
df = df_1.dropna()
value_mapping1 = {33.0: 1, 3467 : 2, 68100.0 : 3}
df['gemhhinc'] = df['gemhhinc'].replace(value_mapping1)
value_mapping2 = {0.0:1,111.0:2, 1212.0:3, 1316.0:4, 1720.0:5}
df['gemeduc'] = df['gemeduc'].replace(value_mapping2)

#Simple EDA
st.subheader('Descriptive Statistics')
st.dataframe(df.drop(['region'], axis=1).describe(), width=800, height=320)

# Load model and preprocessing objects
@st.cache_resource
def load_model_objects():
    model_xgb = joblib.load('model_xgb.joblib')
    scaler = joblib.load('scaler.joblib')
    ohe = joblib.load('ohe.joblib')
    return model_xgb, scaler, ohe

model_xgb, scaler, ohe = load_model_objects()

# Create SHAP explainer
explainer = shap.TreeExplainer(model_xgb)

# User inputs
col1, col2 = st.columns(2)

with col1:
    gemhhinc = st.selectbox('Income level: 1=Low, 2=Medium, 3=High', options=ohe.categories_[0])
    gemeduc = st.radio('Level of completed education: 1=Low, 2=Semi-low, 3=Medium, 4=Semi-High, 5=High', options=ohe.categories_[1])
    nbmedial = st.selectbox('The level of media coverage on entrepreneurs: 1=Low, 2=Semi-low, 3=Medium, 4=Semi-High, 5=High', options=ohe.categories_[2])

with col2:
    region = st.radio('Region: 1=Midde East & Africa, 2=Central & East Asia, 3=Latin America & Caribbean, 4=Europe & N America', options=ohe.categories_[3])
    age = st.number_input('Age', min_value=1, max_value=80, value=1)

# Prediction button
if st.button('Predict bstart 🚀'):
    # Prepare categorical features
    cat_features = pd.DataFrame({'gemhhinc': [gemhhinc], 'gemeduc': [gemeduc], 'nbmedial': [nbmedial], 'region': [region]})
    cat_encoded = pd.DataFrame(ohe.transform(cat_features).todense(), 
                               columns=ohe.get_feature_names_out(['gemhhinc', 'gemeduc', 'nbmedial', 'region']))
    
    # Prepare numerical features
    num_features = pd.DataFrame({
        'age': [age],
    })
    num_scaled = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
    
    # Combine features
    features = pd.concat([num_scaled, cat_encoded], axis=1)
    
    # Make prediction
    predicted_bstart = model_xgb.predict(features)[0]
    
    # Display prediction
    st.metric(label="Predicted likelihood of business start", value=predicted_bstart)
    
    
    # SHAP explanation
    st.subheader('Business start 🤖')
    shap_values = explainer.shap_values(features)
    st_shap(shap.force_plot(explainer.expected_value, shap_values, features), height=400, width=600)
    
    st.markdown("""
    This plot shows how each feature contributes to the predicted price:
    - Blue bars push the likelihood lower
    - Red bars push the likelihood higher
    - The length of each bar indicates the strength of the feature's impact
    """)
