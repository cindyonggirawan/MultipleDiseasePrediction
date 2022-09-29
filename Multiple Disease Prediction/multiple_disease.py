import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

#Loading the saved scalers
diabetes_scaler = pickle.load(open("/Users/cindy/Documents/Kuliah/Multiple Disease Prediction/saved scalers/diabetes_scaler.sav", 'rb'))
parkinsons_scaler = pickle.load(open("/Users/cindy/Documents/Kuliah/Multiple Disease Prediction/saved scalers/parkinsons_scaler.sav", 'rb'))

#Loading the saved models
diabetes_model = pickle.load(open("/Users/cindy/Documents/Kuliah/Multiple Disease Prediction/saved models/diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("/Users/cindy/Documents/Kuliah/Multiple Disease Prediction/saved models/heart_disease_model.sav", 'rb'))
parkinsons_model = pickle.load(open("/Users/cindy/Documents/Kuliah/Multiple Disease Prediction/saved models/parkinsons_model.sav", 'rb'))
breast_cancer_model = pickle.load(open("/Users/cindy/Documents/Kuliah/Multiple Disease Prediction/saved models/breast_cancer_model.sav", 'rb'))

def heart_disease_prediction(input_data):
    #Changing the input_data to numpyarray
    input_data_as_numpy_array = np.asarray(input_data)

    #Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = heart_disease_model.predict(input_data_reshaped)

    return prediction[0]

def breast_cancer_prediction(input_data):
    #Changing the input_data to numpyarray
    input_data_as_numpy_array = np.asarray(input_data)

    #Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = breast_cancer_model.predict(input_data_reshaped)

    return prediction[0]

def diabetes_prediction(input_data):
    #Changing the input_data to numpyarray
    input_data_as_numpy_array = np.asarray(input_data)

    #Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    #Standarize the input_data
    std_data = diabetes_scaler.transform(input_data_reshaped)

    prediction = diabetes_model.predict(std_data)

    return prediction[0]

def parkinsons_prediction(input_data):
    #Changing the input_data to numpyarray
    input_data_as_numpy_array = np.asarray(input_data)

    #Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    #Standarize the input_data
    std_data = parkinsons_scaler.transform(input_data_reshaped)

    prediction = parkinsons_model.predict(std_data)

    return prediction[0]

#Sidebar for navigation
with st.sidebar:
    selected = option_menu(menu_title="Multiple Disease Prediction System",
                           options=["Heart Disease Prediction",
                                    "Breast Cancer Prediction",
                                    "Diabetes Prediction",
                                    "Parkinsons Prediction"],
                           icons=['heart', 'person', 'bandaid', 'activity'],
                           menu_icon="cast",
                           default_index=0)

if (selected == "Heart Disease Prediction"):
    #Page Title
    st.title("Heart Disease Prediction using Logistic Regression")
    
    accuracy = 0.8 * 85.12 + 0.2 * 81.96
    st.info(f"Accuracy score of the model is **{accuracy}%**")
    
    #Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=1, max_value=100, step=1)
    
    with col2:
        select_sex = st.selectbox("Sex", ["Female", "Male"])
        if (select_sex == "Female"):
            sex = 0
        else:
            sex = 1
    
    with col3:
        select_cp = st.selectbox("Chest Pain Types", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
        if (select_cp == "Typical angina"):
            cp = 0
        elif (select_cp == "Atypical angina"):
            cp = 1
        elif (select_cp == "Non-anginal pain"):
            cp = 2
        else:
            cp = 3
    
    with col1:
        trestbps = st.text_input("Resting Blood Pressure in mm Hg")
    
    with col2:
        chol = st.text_input("Serum Cholestoral in mg/dl")
    
    with col3:
        select_fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        if (select_fbs == "False"):
            fbs = 0
        else:
            fbs = 1
    
    with col1:
        select_restecg = st.selectbox("Resting Electrocardiographic results", ["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"])
        if (select_restecg == "Normal"):
            restecg = 0
        elif (select_restecg == "Having ST-T wave abnormality"):
            restecg = 1
        else:
            restecg = 2
    
    with col2:
        thalach = st.text_input("Maximum Heart Rate achieved")
    
    with col3:
        select_exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        if (select_exang == "No"):
            exang = 0
        else:
            exang = 1
    
    with col1:
        oldpeak = st.number_input("ST depression induced by exercise")
    
    with col2:
        select_slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
        if (select_slope == "Upsloping"):
            slope = 0
        elif (select_restecg == "Flat"):
            slope = 1
        else:
            slope = 2
    
    with col3:
        ca = st.slider("Major vessels colored by flourosopy", min_value=0, max_value=3, step=1)
    
    with col1:
        select_thal = st.selectbox("Thal", ["Normal", "Fixed defect", "Reversable defect"])
        if (select_thal == "Normal"):
            thal = 0
        elif (select_thal == "Fixed defect"):
            thal = 1
        else:
            thal = 2
    
    #Creating a button for prediction
    if st.button("Heart Disease Test Result"):
        heart_disease_diagnosis = heart_disease_prediction([int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)])
        
        if (heart_disease_diagnosis == 0):
            st.success("The person does not have a heart disease")
        else:
            st.error("The person has a heart disease")

if (selected == "Breast Cancer Prediction"):
    #Page Title
    st.title("Breast Cancer Prediction using Logistic Regression")
    
    accuracy = 0.8 * 94.06 + 0.2 * 96.49
    st.info(f"Accuracy score of the model is **{accuracy}%**")
    
    #Getting the input data from the user
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mean_radius = st.number_input("Mean Radius")
    
    with col2:
        mean_texture = st.number_input("Mean Texture")
    
    with col3:
        mean_perimeter = st.number_input("Mean Perimeter")
    
    with col4:
        mean_area = st.number_input("Mean Area")
    
    with col5:
        mean_smoothness = st.number_input("Mean Smoothness")
    
    with col1:
        mean_compactness = st.number_input("Mean Compactness")
    
    with col2:
        mean_concavity = st.number_input("Mean Concavity")
    
    with col3:
        mean_concave_points = st.number_input("Mean Concave Points")
    
    with col4:
        mean_symmetry = st.number_input("Mean Symmetry")
    
    with col5:
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension")
    
    with col1:
        radius_error = st.number_input("Radius Error")
    
    with col2:
        texture_error = st.number_input("Texture Error")
    
    with col3:
        perimeter_error = st.number_input("Perimeter Error")
    
    with col4:
        area_error = st.number_input("Area Error")
    
    with col5:
        smoothness_error = st.number_input("Smoothness Error")
    
    with col1:
        compactness_error = st.number_input("Compactness Error")
    
    with col2:
        concavity_error = st.number_input("Concavity Error")
    
    with col3:
        concave_points_error = st.number_input("Concave Points Error")
    
    with col4:
        symmetry_error = st.number_input("Symmetry Error")
    
    with col5:
        fractal_dimension_error = st.number_input("Fractal Dimension Error")
    
    with col1:
        worst_radius = st.number_input("Worst Radius")
    
    with col2:
        worst_texture = st.number_input("Worst Texture")
    
    with col3:
        worst_perimeter = st.number_input("Worst Perimeter")
    
    with col4:
        worst_area = st.number_input("Worst Area")
    
    with col5:
        worst_smoothness = st.number_input("Worst Smoothness")
    
    with col1:
        worst_compactness = st.number_input("Worst Compactness")
    
    with col2:
        worst_concavity = st.number_input("Worst Concavity")
    
    with col3:
        worst_concave_points = st.number_input("Worst Concave Points")
    
    with col4:
        worst_symmetry = st.number_input("Worst Symmetry")
    
    with col5:
        worst_fractal_dimension = st.number_input("Worst Fractal Dimension")
    
    #Creating a button for prediction
    if st.button("Breast Cancer Test Result"):
        breast_cancer_diagnosis = breast_cancer_prediction([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension])
        
        if (breast_cancer_diagnosis == 0):
            st.success("The person's breast cancer is malignant")
        else:
            st.error("The person's breast cancer is benign")
    
if (selected == "Diabetes Prediction"):
    #Page Title
    st.title("Diabetes Prediction using Support Vector Machine")
        
    accuracy = 0.8 * 78.66 + 0.2 * 77.27
    st.info(f"Accuracy score of the model is **{accuracy}%**")
        
    #Getting the input data from the user
    col1, col2, col3 = st.columns(3)
        
    with col1:
        Pregnancies = st.slider("Number of Pregnancies", min_value=0, max_value=20, step=1)
            
    with col2:
        Glucose = st.slider("Glucose Level", min_value=0, max_value=200, step=1)
        
    with col3:
        BloodPressure = st.slider("Blood Pressure Value", min_value=0, max_value=130, step=1)
        
    with col1:
        SkinThickness = st.slider("Skin Thickness Value", min_value=0, max_value=100, step=1)
        
    with col2:
        Insulin = st.text_input("Insulin Level")
        
    with col3:
        Bmi = st.number_input("BMI Value")
        
    with col1:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function Value")
        
    with col2:
        Age = st.slider("Age", min_value=1, max_value=100, step=1)
        
    #Creating a button for prediction
    if st.button("Diabetes Test Result"):
        diabetes_diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Bmi, DiabetesPedigreeFunction, Age])

        if (diabetes_diagnosis == 0):
            st.success("The person does not have diabetes")
        else:
            st.error("The person has diabetes")
    
if (selected == "Parkinsons Prediction"):
    #Page Title
    st.title("Parkinsons Prediction using Support Vector Machine")
    
    accuracy = 0.8 * 88.46 + 0.2 * 87.17
    st.info(f"Accuracy score of the model is **{accuracy}%**")
    
    #Getting the input data from the user
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        Fo = st.number_input("MDVP:Fo(Hz)")
    
    with col2:
        Fhi = st.number_input("MDVP:Fhi(Hz)")
    
    with col3:
        Flo = st.number_input("MDVP:Flo(Hz)")
    
    with col4:
        Jitter = st.number_input("MDVP:Jitter(%)")
    
    with col5:
        JitterAbs = st.number_input("MDVP:Jitter(Abs)")
    
    with col1:
        Rap = st.number_input("MDVP:RAP")
    
    with col2:
        Ppq = st.number_input("MDVP:PPQ")
    
    with col3:
        Ddp = st.number_input("Jitter:DDP")
    
    with col4:
        Shimmer = st.number_input("MDVP:Shimmer")
    
    with col5:
        ShimmerdB = st.number_input("MDVP:Shimmer(dB)")
    
    with col1:
        Apq3 = st.number_input("Shimmer:APQ3")
    
    with col2:
        Apq5 = st.number_input("Shimmer:APQ5")
    
    with col3:
        Apq = st.number_input("MDVP:APQ")
    
    with col4:
        Dda = st.number_input("Shimmer:DDA")
    
    with col5:
        Nhr = st.number_input("NHR")
    
    with col1:
        Hnr = st.number_input("HNR")
    
    with col2:
        Rpde = st.number_input("RPDE")
    
    with col3:
        Dfa = st.number_input("DFA")
    
    with col4:
        spread1 = st.number_input("spread1")
    
    with col5:
        spread2 = st.number_input("spread2")
    
    with col1:
        d2 = st.number_input("D2")
    
    with col2:
        Ppe = st.number_input("PPE")
        
    #Creating a button for prediction
    if st.button("Parkinsons Test Result"):
        parkinsons_diagnosis = parkinsons_prediction([Fo, Fhi, Flo, Jitter, JitterAbs, Rap, Ppq, Ddp, Shimmer, ShimmerdB, Apq3, Apq5, Apq, Dda, Nhr, Hnr, Rpde, Dfa, spread1, spread2, d2, Ppe])

        if (parkinsons_diagnosis == 0):
            st.success("The person does not have Parkinson's disease")
        else:
            st.error("The person has Parkinson's disease")