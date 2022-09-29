import numpy as np
import pickle

#Loading the saved scaler
loaded_scaler = pickle.load(open("/Users/cindy/Documents/Kuliah/Machine Learning Project/diabetes_scaler.sav", 'rb'))
#Loading the saved model
loaded_model = pickle.load(open("/Users/cindy/Documents/Kuliah/Machine Learning Project/trained_diabetes_model.sav", 'rb')) #read binary

input_data = (5,166,72,19,175,25.8,0.587,51) #list

#Changing the input_data to numpyarray
input_data_as_numpy_array = np.asarray(input_data)

#Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#Standarize the input_data
std_data = loaded_scaler.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")