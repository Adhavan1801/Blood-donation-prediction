import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# Load the trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

def predict_donation(recency, frequency, monetary, time):
    # Prepare input data as a dataframe to match the column names used during training
    input_data = pd.DataFrame([[recency, frequency, monetary, time]], 
                              columns=['Recency', 'Frequency', 'Monetary', 'Time'])
    
    # Scale the input
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    # Format the result
    result_text = "Yes, likely to donate blood." if prediction == 1 else "No, unlikely to donate blood."
    confidence = f"Confidence: {max(probabilities) * 100:.2f}%"
    
    return f"{result_text}\n{confidence}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩸 Blood Donation Prediction")
    gr.Markdown("Predict whether a person is likely to donate blood based on their past donation history (RFMTC model).")
    
    with gr.Row():
        with gr.Column():
            recency = gr.Number(label="Recency (months since last donation)", value=2)
            frequency = gr.Number(label="Frequency (total number of donations)", value=5)
            monetary = gr.Number(label="Monetary (total blood donated in c.c.)", value=1250)
            time = gr.Number(label="Time (months since first donation)", value=28)
            
            predict_btn = gr.Button("Predict", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Prediction Result", lines=3)
            
    predict_btn.click(
        fn=predict_donation,
        inputs=[recency, frequency, monetary, time],
        outputs=output_text
    )
    
    gr.Markdown("---")
    gr.Markdown("**Features Meaning:**")
    gr.Markdown("- **Recency:** months since last donation")
    gr.Markdown("- **Frequency:** total number of donation")
    gr.Markdown("- **Monetary:** total blood donated in c.c. (Frequency x 250)")
    gr.Markdown("- **Time:** months since first donation")

if __name__ == "__main__":
    demo.launch()
