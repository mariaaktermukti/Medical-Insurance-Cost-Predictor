import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load trained model

with open("insurance_rf_pipeline.pkl", "rb") as file:
    model = pickle.load(file)


# Prediction function
def predict_insurance(age, sex, bmi, children, smoker, region):
    
    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region
    ]], columns=[
        "age", "sex", "bmi", "children", "smoker", "region"
    ])
    
    prediction = model.predict(input_df)[0]
    
    return f"Predicted Insurance Cost: ${prediction:,.2f}"


# Gradio Inputs
inputs = [
    gr.Number(label="Age", value=25),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Number(label="BMI", value=25.0),
    gr.Slider(0, 5, step=1, label="Number of Children"),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Dropdown(
        ["southeast", "southwest", "northeast", "northwest"],
        label="Region"
    )
]


# Gradio Interface
app = gr.Interface(
    fn=predict_insurance,
    inputs=inputs,
    outputs="text",
    title="Medical Insurance Cost Predictor",
    description="Predicts medical insurance cost using a trained Random Forest model"
)

# Launch App
app.launch(share=True)
