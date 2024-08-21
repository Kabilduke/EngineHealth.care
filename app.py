import pickle
import streamlit as stream
import numpy as np
import pandas as pd

#importing the model
with open("svm_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

#importing the scaler model
with open("scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

log_df = pd.DataFrame(columns=['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp', 'Prediction'])

def EngineHealth_predict(input_data, model, scaler, log_df):

    columns = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
    input_df = pd.DataFrame([input_data], columns=columns)

    input_array = np.array(input_df).reshape(1, -1)

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)


    if prediction[0] == 0:
        health_status = "Engine is in Good Health!"
    else:
        health_status = "Engine is not in Good Health"

    input_data.append(prediction[0])
    log_df.loc[len(log_df)] = input_data

    return health_status, log_df

stream.title("Engine Health Prediction")

engine_rpm = stream.number_input("Enter Engine rpm:", min_value=0.0, step=0.1)
lub_oil_pressure = stream.number_input("Enter Lub oil pressure:", min_value=0.0, step=0.1)
fuel_pressure = stream.number_input("Enter Fuel pressure:", min_value=0.0, step=0.1)
coolant_pressure = stream.number_input("Enter Coolant pressure:", min_value=0.0, step=0.1)
lub_oil_temp = stream.number_input("Enter Lub oil temp:", min_value=0.0, step=0.1)
coolant_temp = stream.number_input("Enter Coolant temp:", min_value=0.0, step=0.1)


if stream.button('Predict'):
    input_data = [
        engine_rpm,
        lub_oil_pressure,
        fuel_pressure,
        coolant_pressure,
        lub_oil_temp,
        coolant_temp
    ]
    
    health_status, log_df = EngineHealth_predict(input_data, model, scaler, log_df)
    
    stream.write(f"Prediction: {health_status}")
    
    # Display the log DataFrame
    stream.write("Log of Predictions:")
    stream.dataframe(log_df)