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
    prob_donate  = float(probs[1]) * 100
    prob_not     = float(probs[0]) * 100

    if prediction == 1:
        color     = "#2e7d32"
        bg_color  = "#e8f5e9"
        border_color = "#a5d6a7"
        icon      = "✅"
        title     = "Likely to Donate"
        msg       = "This donor profile indicates a <strong>high probability</strong> of returning to donate blood."
        bar_color = "#2e7d32"
    else:
        color     = "#c62828"
        bg_color  = "#ffebee"
        border_color = "#ef9a9a"
        icon      = "❌"
        title     = "Unlikely to Donate"
        msg       = "This donor profile indicates a <strong>low probability</strong> of returning to donate blood."
        bar_color = "#c62828"

    return f"""
<div style="font-family:'Inter',sans-serif;">

  <!-- Result Card -->
  <div style="background:{bg_color}; border:1px solid {border_color};
              padding:24px 28px; border-radius:12px; margin-bottom:16px;">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
      <span style="font-size:1.6rem;">{icon}</span>
      <h3 style="color:{color}; margin:0; font-size:1.25rem; font-weight:800; letter-spacing:-0.5px;">
        {title}
      </h3>
    </div>
    <p style="margin:0 0 16px; font-size:0.92rem; color:#333; line-height:1.7;">
      {msg}
    </p>

    <!-- Confidence Bar -->
    <div style="background:rgba(0,0,0,0.06); border-radius:8px; overflow:hidden; height:10px; margin-bottom:8px;">
      <div style="width:{confidence:.1f}%; height:100%; background:{bar_color}; border-radius:8px; transition:width 0.5s ease;"></div>
    </div>
    <p style="margin:0; font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#555;">
      Confidence: <strong>{confidence:.2f}%</strong>
    </p>
  </div>

  <!-- Probability Breakdown -->
  <div style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:12px; padding:20px 24px;">
    <p style="margin:0 0 14px; font-size:0.82rem; font-weight:700; color:#424242; text-transform:uppercase; letter-spacing:1px;">
      Probability Breakdown
    </p>
    <div style="display:flex; gap:12px;">
      <div style="flex:1; background:#e8f5e9; border-radius:8px; padding:14px 16px; text-align:center;">
        <p style="margin:0 0 4px; font-size:0.72rem; color:#616161; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">
          Will Donate
        </p>
        <p style="margin:0; font-size:1.4rem; font-weight:800; color:#2e7d32; font-family:'JetBrains Mono',monospace;">
          {prob_donate:.1f}%
        </p>
      </div>
      <div style="flex:1; background:#ffebee; border-radius:8px; padding:14px 16px; text-align:center;">
        <p style="margin:0 0 4px; font-size:0.72rem; color:#616161; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">
          Won't Donate
        </p>
        <p style="margin:0; font-size:1.4rem; font-weight:800; color:#c62828; font-family:'JetBrains Mono',monospace;">
          {prob_not:.1f}%
        </p>
      </div>
    </div>
  </div>

  <!-- Input Summary -->
  <div style="margin-top:16px; background:#F5F5F5; border-radius:10px; padding:14px 20px;">
    <p style="margin:0 0 8px; font-size:0.72rem; font-weight:700; color:#9E9E9E; text-transform:uppercase; letter-spacing:1px;">
      Input Summary
    </p>
    <div style="display:flex; gap:10px; flex-wrap:wrap;">
      <span style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:6px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#333;">
        R={recency}
      </span>
      <span style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:6px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#333;">
        F={frequency}
      </span>
      <span style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:6px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#333;">
        M={monetary}
      </span>
      <span style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:6px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#333;">
        T={time}
      </span>
    </div>
  </div>
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
        font-weight: 600 !important;
    }

    /* Premium input styling */
    input[type="number"], input[type="text"], .gradio-container input, .svelte-1gfkn6j {
        border: 2px solid #E0E0E0 !important;
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #0A0A0A !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease !important;
    }
    input[type="number"]:focus, input[type="text"]:focus {
        border-color: #dd6b3b !important;
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(221,107,59,0.1) !important;
    }
    input[type="number"]:hover, input[type="text"]:hover {
        border-color: #BDBDBD !important;
    }

    /* Premium button styling */
    .primary.svelte-cmf5ev, button.primary {
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(221,107,59,0.3) !important;
        transition: all 0.25s ease !important;
    }
    .primary.svelte-cmf5ev:hover, button.primary:hover {
        box-shadow: 0 6px 20px rgba(221,107,59,0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Input card group styling */
    .group.svelte-1kp1fvl, .gradio-group {
        border-radius: 16px !important;
        border: 1px solid #E0E0E0 !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
        padding: 24px !important;
    }
"""

# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks() as demo:

    # ── Header ──
    gr.HTML("""
    <div style="padding:16px 8px 10px; font-family:'Inter',sans-serif;">
      <p style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:600;
                color:#dd6b3b; text-transform:uppercase; letter-spacing:2px; margin:0 0 4px;">
        Machine Learning Project
      </p>
      <h1 style="font-size:1.8rem; font-weight:900; letter-spacing:-1px; color:#0A0A0A;
                 margin:0 0 6px; line-height:1.15;">
        Blood Donation Prediction
      </h1>
      <p style="color:#616161; font-size:0.9rem; margin:0 0 10px; line-height:1.5;">
        Predicting blood donor re-donation behavior using the UCI RFMTC framework,
        SMOTE oversampling &amp; a tuned Decision Tree classifier.
      </p>
      <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px;">
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:6px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">Decision Tree</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:6px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">SMOTE</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:6px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">GridSearchCV</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:6px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">RFMTC Dataset</span>
        <span style="background:rgba(221,107,59,0.1); color:#dd6b3b;
                     border:1px solid rgba(221,107,59,0.3); border-radius:6px;
                     padding:3px 13px; font-size:0.78rem; font-weight:700;
                     font-family:'JetBrains Mono',monospace;">scikit-learn</span>
      </div>
      <div style="border-top:1px solid #EEEEEE;"></div>
    </div>
    """)

    # ── Main layout ──
    with gr.Row():
        # Left: Inputs
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML("""
                <div style="text-align:center; margin-bottom:20px; margin-top:4px;">
                  <div style="display:inline-flex; align-items:center; gap:8px; margin-bottom:6px;">
                    <span style="font-size:1.3rem;">🩸</span>
                    <h3 style="margin:0; color:#0A0A0A; font-family:'Inter',sans-serif; font-size:1.05rem; font-weight:800; letter-spacing:-0.3px;">
                      Enter Donor Information
                    </h3>
                  </div>
                  <p style="margin:0; color:#9E9E9E; font-size:0.78rem; font-family:'Inter',sans-serif;">
                    Fill in the RFMTC values below to predict donor behavior
                  </p>
                </div>
                """)

                # Feature info cards above inputs
                gr.HTML("""
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:16px; font-family:'Inter',sans-serif;">
                  <div style="background:#FFF3E0; border-radius:8px; padding:10px 14px;">
                    <p style="margin:0 0 2px; font-size:0.65rem; font-weight:700; color:#dd6b3b; text-transform:uppercase; letter-spacing:0.8px;">Recency</p>
                    <p style="margin:0; font-size:0.75rem; color:#424242;">Months since last donation</p>
                  </div>
                  <div style="background:#FFF3E0; border-radius:8px; padding:10px 14px;">
                    <p style="margin:0 0 2px; font-size:0.65rem; font-weight:700; color:#dd6b3b; text-transform:uppercase; letter-spacing:0.8px;">Frequency</p>
                    <p style="margin:0; font-size:0.75rem; color:#424242;">Total number of donations</p>
                  </div>
                  <div style="background:#FFF3E0; border-radius:8px; padding:10px 14px;">
                    <p style="margin:0 0 2px; font-size:0.65rem; font-weight:700; color:#dd6b3b; text-transform:uppercase; letter-spacing:0.8px;">Monetary</p>
                    <p style="margin:0; font-size:0.75rem; color:#424242;">Total blood donated (c.c.)</p>
                  </div>
                  <div style="background:#FFF3E0; border-radius:8px; padding:10px 14px;">
                    <p style="margin:0 0 2px; font-size:0.65rem; font-weight:700; color:#dd6b3b; text-transform:uppercase; letter-spacing:0.8px;">Time</p>
                    <p style="margin:0; font-size:0.75rem; color:#424242;">Months since first donation</p>
                  </div>
                </div>
                """)

                with gr.Row():
                    recency   = gr.Number(label="Recency (months since last)", value=2, minimum=0)
                    frequency = gr.Number(label="Frequency (total donations)", value=5, minimum=1)
                with gr.Row():
                    monetary  = gr.Number(label="Monetary (total c.c.)",       value=1250, minimum=250)
                    time      = gr.Number(label="Time (months since first)",   value=28, minimum=1)
                btn       = gr.Button("🔬  Run Prediction", variant="primary", size="lg")

            gr.HTML("""
            <div style="margin-top:12px; padding:12px 16px; background:#F5F5F5; border-radius:10px; font-family:'Inter',sans-serif;">
              <p style="margin:0; font-size:0.82rem; color:#616161; line-height:1.6;">
                💡 <strong style="color:#0A0A0A;">Tip:</strong> Monetary = Frequency × 250 c.c.
              </p>
            </div>
            """)

        # Right: Output
        with gr.Column(scale=1):
            out_html = gr.HTML("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                        min-height:320px; font-family:'Inter',sans-serif; color:#BDBDBD;">
              <span style="font-size:3rem; margin-bottom:12px; opacity:0.5;">🔬</span>
              <p style="margin:0; font-size:0.95rem; font-weight:600;">Waiting for prediction…</p>
              <p style="margin:4px 0 0; font-size:0.8rem; color:#BDBDBD;">
                Click <strong style="color:#dd6b3b;">Run Prediction</strong> to see the result
              </p>
            </div>
            """)

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
