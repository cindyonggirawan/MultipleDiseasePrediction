import numpy as np
import pickle
import streamlit as st

#Loading the saved scaler
loaded_scaler = pickle.load(open("/Users/cindy/Documents/Kuliah/Machine Learning Project/diabetes_scaler.sav", 'rb'))
#Loading the saved model
loaded_model = pickle.load(open("/Users/cindy/Documents/Kuliah/Machine Learning Project/trained_diabetes_model.sav", 'rb')) #read binary

#Create a function for prediction
def diabetes_prediction(input_data):
    #Changing the input_data to numpyarray
    input_data_as_numpy_array = np.asarray(input_data)

    #Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    #Standarize the input_data
    std_data = loaded_scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

def main():
    #Giving a title
    st.title("Diabetes Prediction Web App")
    
    #Getting the input data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    Bmi = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the Person")
    
    #Code for prediction
    Diagnosis = ""
    
    #Creating a button for prediction
    if st.button("Diabetes Test Result"):
        Diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Bmi, DiabetesPedigreeFunction, Age])
    
    st.success(Diagnosis)
    
if __name__ == '__main__':
    main()