import gradio as gr
import joblib
import pandas as pd
import numpy as np

# ── Load Model and Scaler ─────────────────────────────────────────────
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# ── Prediction Logic ──────────────────────────────────────────────────
def predict_donation(recency, frequency, monetary, time):
    input_data = pd.DataFrame([[recency, frequency, monetary, time]], 
                              columns=['Recency', 'Frequency', 'Monetary', 'Time'])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    # Generate stylized HTML output instead of plain text
    prob_percentage = max(probabilities) * 100
    if prediction == 1:
        color = "#2e7d32" # green
        bg_color = "#e8f5e9"
        icon = "✅"
        result_title = "Likely to Donate"
        msg = "This donor profile indicates a high probability of returning to donate blood."
    else:
        color = "#c62828" # red
        bg_color = "#ffebee"
        icon = "❌"
        result_title = "Unlikely to Donate"
        msg = "This donor profile indicates a low probability of returning to donate blood."

    html_out = f"""
    <div style="background-color: {bg_color}; border-left: 6px solid {color}; padding: 16px; border-radius: 4px; margin-top: 10px;">
        <h3 style="color: {color}; margin: 0 0 8px 0; font-family: 'Inter', sans-serif;">
            {icon} {result_title}
        </h3>
        <p style="margin: 0 0 8px 0; font-family: 'Inter', sans-serif; font-size: 0.95rem; color: #333;">
            {msg}
        </p>
        <p style="margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #555;">
            <strong>Confidence:</strong> {prob_percentage:.2f}%
        </p>
    </div>
    """
    return html_out

# ── Portfolio-matched theme ───────────────────────────────────────────
portfolio_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#FAFAFA",
    body_background_fill_dark="#FAFAFA",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#F5F5F5",
    background_fill_secondary_dark="#F5F5F5",
    body_text_color="#0A0A0A",
    body_text_color_dark="#0A0A0A",
    body_text_color_subdued="#616161",
    body_text_color_subdued_dark="#616161",
    border_color_primary="#E0E0E0",
    border_color_primary_dark="#E0E0E0",
    button_primary_background_fill="#dd6b3b",
    button_primary_background_fill_dark="#dd6b3b",
    button_primary_background_fill_hover="#C26345",
    button_primary_background_fill_hover_dark="#C26345",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_background_fill_hover="#FFF3E0",
    button_secondary_background_fill_hover_dark="#FFF3E0",
    button_secondary_text_color="#0A0A0A",
    button_secondary_text_color_dark="#0A0A0A",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#E0E0E0",
    input_border_color_dark="#E0E0E0",
    input_border_color_focus="#dd6b3b",
    input_border_color_focus_dark="#dd6b3b",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#EEEEEE",
    block_border_color_dark="#EEEEEE",
    block_label_background_fill="#F5F5F5",
    block_label_background_fill_dark="#F5F5F5",
    block_label_text_color="#424242",
    block_label_text_color_dark="#424242",
)

# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(
    title="Blood Donation Prediction — RFMTC Pipeline",
    theme=portfolio_theme,
    css="""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
        body, .gradio-container { background:#FAFAFA !important; }
        footer { display:none !important; }
        /* Box styling overrides if needed */
        .gradio-container { font-family: 'Inter', sans-serif !important; }
    """
) as demo:

    # ── Header ──
    gr.HTML("""
    <div style="padding:32px 8px 20px; font-family:'Inter',sans-serif;">
      <p style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:600;
                color:#dd6b3b; text-transform:uppercase; letter-spacing:2px; margin:0 0 8px;">
        Machine Learning Pipeline
      </p>
      <h1 style="font-size:2rem; font-weight:900; letter-spacing:-1px; color:#0A0A0A;
                 margin:0 0 8px; line-height:1.15;">
        Blood Donation Prediction
      </h1>
      <p style="color:#616161; font-size:0.95rem; margin:0 0 16px; line-height:1.6;">
        Predicting blood donor re-donation behavior using the UCI RFMTC framework and SMOTE-balanced Decision Trees.
      </p>
      <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:20px;">
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b; border:1px solid rgba(221,107,59,0.3); border-radius:999px; padding:3px 13px; font-size:0.78rem; font-weight:700; font-family:'JetBrains Mono',monospace;">Decision Tree</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b; border:1px solid rgba(221,107,59,0.3); border-radius:999px; padding:3px 13px; font-size:0.78rem; font-weight:700; font-family:'JetBrains Mono',monospace;">SMOTE</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b; border:1px solid rgba(221,107,59,0.3); border-radius:999px; padding:3px 13px; font-size:0.78rem; font-weight:700; font-family:'JetBrains Mono',monospace;">GridSearchCV</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b; border:1px solid rgba(221,107,59,0.3); border-radius:999px; padding:3px 13px; font-size:0.78rem; font-weight:700; font-family:'JetBrains Mono',monospace;">RFMTC Data</span>
      </div>
      <div style="border-top:1px solid #EEEEEE;"></div>
    </div>
    """)

    # ── Main layout ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("**Enter Donor Information**")
            recency = gr.Number(label="Recency (months since last donation)", value=2)
            frequency = gr.Number(label="Frequency (total donations)", value=5)
            monetary = gr.Number(label="Monetary (total blood donated in c.c.)", value=1250)
            time = gr.Number(label="Time (months since first donation)", value=28)
            
            predict_btn = gr.Button("Run Prediction", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("**Prediction Result**")
            out_html = gr.HTML('<div style="color:#999; font-family:\'Inter\', sans-serif; padding:10px 0;">Click "Run Prediction" to see the result...</div>')
            
            gr.Markdown("""
            ---
            **Features Explained:**
            - **Recency:** Months since the donor's last blood donation.
            - **Frequency:** Total number of past donations.
            - **Monetary:** Total blood donated in c.c. (Frequency × 250).
            - **Time:** Months since the donor's very first donation.
            """)

    # Setup examples
    gr.Examples(
        examples=[
            [2, 50, 12500, 98],
            [12, 1, 250, 12],
            [21, 3, 750, 38],
            [2, 2, 500, 4]
        ],
        inputs=[recency, frequency, monetary, time],
        label="Example Profiles"
    )

    predict_btn.click(fn=predict_donation, inputs=[recency, frequency, monetary, time], outputs=out_html)

    # ── Footer ──
    gr.HTML("""
    <div style="text-align:center; padding:16px 0 8px; font-family:'Inter',sans-serif;
                color:#9E9E9E; font-size:0.8rem; border-top:1px solid #EEEEEE; margin-top:16px;">
      Built by&nbsp;
      <a href="https://github.com/adhavan1801" target="_blank"
         style="color:#dd6b3b; font-weight:600; text-decoration:none;">Adhavan U S</a>
      &nbsp;&middot;&nbsp;
      Decision Tree Classification Pipeline
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
