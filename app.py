import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load trained model
with open("insurance_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

def predict_cost(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([{
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(input_df)[0]
    return f"Predicted Insurance Cost: ${prediction:,.2f}"

app = gr.Interface(
    fn=predict_cost,
    inputs=[
        gr.Number(label="Age", value=30),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="BMI", value=22.0),
        gr.Slider(0, 5, step=1, label="Number of Children"),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Dropdown(
            ["northeast", "northwest", "southeast", "southwest"],
            label="Region"
        )
    ],
    outputs="text",
    title="Medical Insurance Cost Predictor",
    description="Predicts medical insurance cost using a trained Random Forest model"
)

app.launch()
