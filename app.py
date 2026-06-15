# -*- coding: utf-8 -*-
"""
Blood Donation Prediction — Hugging Face Spaces Inference App
Adhavan U S
"""

import gradio as gr
import joblib
import pandas as pd
import numpy as np

# ── Load Model and Scaler ─────────────────────────────────────────────
model  = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# ── Prediction Logic ──────────────────────────────────────────────────
def predict_donation(recency, frequency, monetary, time):
    input_data   = pd.DataFrame([[recency, frequency, monetary, time]],
                                columns=['Recency', 'Frequency', 'Monetary', 'Time'])
    scaled_input = scaler.transform(input_data)
    prediction   = model.predict(scaled_input)[0]
    probs        = model.predict_proba(scaled_input)[0]
    confidence   = float(max(probs)) * 100

    if prediction == 1:
        color     = "#2e7d32"
        bg_color  = "#e8f5e9"
        icon      = "✅"
        title     = "Likely to Donate"
        msg       = "This donor profile indicates a high probability of returning to donate blood."
    else:
        color     = "#c62828"
        bg_color  = "#ffebee"
        icon      = "❌"
        title     = "Unlikely to Donate"
        msg       = "This donor profile indicates a low probability of returning to donate blood."

    return f"""
<div style="background:{bg_color}; border-left:6px solid {color};
            padding:20px 24px; border-radius:6px; margin-top:8px;
            font-family:'Inter',sans-serif;">
  <h3 style="color:{color}; margin:0 0 10px; font-size:1.2rem; font-weight:800;">
    {icon}&nbsp; {title}
  </h3>
  <p style="margin:0 0 12px; font-size:0.95rem; color:#333; line-height:1.6;">
    {msg}
  </p>
  <p style="margin:0; font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#555;">
    <strong>Confidence:</strong> {confidence:.2f}%
  </p>
</div>
"""

# ── Portfolio-matched theme (identical to GhostMTFormer) ──────────────
portfolio_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font     =[gr.themes.GoogleFont("Inter"),          "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill                      = "#FAFAFA",
    body_background_fill_dark                 = "#FAFAFA",
    background_fill_primary                   = "#FFFFFF",
    background_fill_primary_dark              = "#FFFFFF",
    background_fill_secondary                 = "#F5F5F5",
    background_fill_secondary_dark            = "#F5F5F5",
    body_text_color                           = "#0A0A0A",
    body_text_color_dark                      = "#0A0A0A",
    body_text_color_subdued                   = "#616161",
    body_text_color_subdued_dark              = "#616161",
    border_color_primary                      = "#E0E0E0",
    border_color_primary_dark                 = "#E0E0E0",
    button_primary_background_fill            = "#dd6b3b",
    button_primary_background_fill_dark       = "#dd6b3b",
    button_primary_background_fill_hover      = "#C26345",
    button_primary_background_fill_hover_dark = "#C26345",
    button_primary_text_color                 = "#FFFFFF",
    button_primary_text_color_dark            = "#FFFFFF",
    button_secondary_background_fill          = "#FFFFFF",
    button_secondary_background_fill_dark     = "#FFFFFF",
    button_secondary_background_fill_hover    = "#FFF3E0",
    button_secondary_background_fill_hover_dark="#FFF3E0",
    button_secondary_text_color               = "#0A0A0A",
    button_secondary_text_color_dark          = "#0A0A0A",
    input_background_fill                     = "#FFFFFF",
    input_background_fill_dark                = "#FFFFFF",
    input_border_color                        = "#E0E0E0",
    input_border_color_dark                   = "#E0E0E0",
    input_border_color_focus                  = "#dd6b3b",
    input_border_color_focus_dark             = "#dd6b3b",
    block_background_fill                     = "#FFFFFF",
    block_background_fill_dark                = "#FFFFFF",
    block_border_color                        = "#EEEEEE",
    block_border_color_dark                   = "#EEEEEE",
    block_label_background_fill               = "#F5F5F5",
    block_label_background_fill_dark          = "#F5F5F5",
    block_label_text_color                    = "#424242",
    block_label_text_color_dark               = "#424242",
)

CUSTOM_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    body, .gradio-container { background:#FAFAFA !important; }
    footer { display:none !important; }

    /* Fix: Input labels solid black */
    label, .label-wrap span, .gradio-container label span,
    .gradio-container .block label span {
        color: #0A0A0A !important;
        font-weight: 500 !important;
    }

    /* Fix: Example table cells white background with black text */
    .examples-table td, .examples-table th,
    table td, table th,
    .gallery td, .gallery th,
    [class*="example"] td, [class*="example"] th,
    .svelte-1wj0ocy td, .svelte-1wj0ocy th {
        background-color: #FFFFFF !important;
        color: #0A0A0A !important;
    }
    .examples-table tr:hover td,
    table tr:hover td {
        background-color: #FFF3E0 !important;
    }

    /* Fix: Visible input boxes */
    input[type="number"], input[type="text"], .gradio-container input, .svelte-1gfkn6j {
        border: 1px solid #E0E0E0 !important;
        background-color: #FFFFFF !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    input[type="number"]:focus, input[type="text"]:focus {
        border-color: #dd6b3b !important;
        outline: none !important;
    }
"""

# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks() as demo:

    # ── Header ──
    gr.HTML("""
    <div style="padding:32px 8px 20px; font-family:'Inter',sans-serif;">
      <p style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:600;
                color:#dd6b3b; text-transform:uppercase; letter-spacing:2px; margin:0 0 8px;">
        Machine Learning Project
      </p>
      <h1 style="font-size:2rem; font-weight:900; letter-spacing:-1px; color:#0A0A0A;
                 margin:0 0 8px; line-height:1.15;">
        Blood Donation Prediction
      </h1>
      <p style="color:#616161; font-size:0.95rem; margin:0 0 16px; line-height:1.6;">
        Predicting blood donor re-donation behavior using the UCI RFMTC framework,
        SMOTE oversampling &amp; a tuned Decision Tree classifier.
      </p>
      <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:20px;">
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:999px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">Decision Tree</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:999px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">SMOTE</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:999px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">GridSearchCV</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:999px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">RFMTC Dataset</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:999px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">scikit-learn</span>
      </div>
      <div style="border-top:1px solid #EEEEEE;"></div>
    </div>
    """)

    # ── Main layout ──
    with gr.Row(equal_height=True):
        # Left: Inputs
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML("<h3 style='text-align: center; margin-bottom: 15px; margin-top: 5px; color: #0A0A0A; font-family: \"Inter\", sans-serif;'>Enter Donor Information</h3>")
                recency   = gr.Number(label="Recency — months since last donation",   value=2,    minimum=0)
                frequency = gr.Number(label="Frequency — total donations",             value=5,    minimum=1)
                monetary  = gr.Number(label="Monetary — total blood donated (c.c.)",  value=1250, minimum=250)
                time      = gr.Number(label="Time — months since first donation",      value=28,   minimum=1)
                btn       = gr.Button("Run Prediction", variant="primary", size="lg")

            gr.Markdown("""
**How to use**
1. Enter the donor's RFMTC values above
2. Click **Run Prediction**
3. See the result and confidence score

> Monetary = Frequency x 250 c.c.
            """)

        # Right: Output
        with gr.Column(scale=1):
            out_html = gr.HTML(
                '<div style="color:#9E9E9E; font-family:\'Inter\',sans-serif; '
                'padding:20px 0; font-size:0.9rem;">'
                'Click <strong>Run Prediction</strong> to see the result…'
                '</div>'
            )
            gr.Markdown("""
---
**Feature Reference**

| Feature | Meaning |
|---------|---------|
| **Recency** | Months since last donation |
| **Frequency** | Total number of donations |
| **Monetary** | Total blood donated in c.c. |
| **Time** | Months since first donation |
| **Target** | Donated in March 2007? |
            """)

    # ── Example Profiles ──
    gr.Examples(
        examples=[
            [2,  50, 12500, 98],
            [12,  1,   250, 12],
            [21,  3,   750, 38],
            [2,   2,   500,  4],
        ],
        inputs=[recency, frequency, monetary, time],
        label="Example Donor Profiles",
    )

    btn.click(fn=predict_donation,
              inputs=[recency, frequency, monetary, time],
              outputs=out_html)

    # ── Footer ──
    gr.HTML("""
    <div style="text-align:center; padding:16px 0 8px; font-family:'Inter',sans-serif;
                color:#9E9E9E; font-size:0.8rem; border-top:1px solid #EEEEEE; margin-top:16px;">
      Built by&nbsp;
      <a href="https://github.com/adhavan1801" target="_blank"
         style="color:#dd6b3b; font-weight:600; text-decoration:none;">Adhavan U S</a>
      &nbsp;&middot;&nbsp;
      Decision Tree · SMOTE · GridSearchCV · scikit-learn
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        theme=portfolio_theme,
        css=CUSTOM_CSS,
    )
