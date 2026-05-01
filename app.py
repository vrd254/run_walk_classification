import gradio as gr
import joblib
import pandas as pd

model         = joblib.load("rf_model.pkl")
scaler        = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

LABELS = {0: "🚶 Walking", 1: "🏃 Running"}

def predict_activity(*sensor_values):
    input_df = pd.DataFrame([list(sensor_values)], columns=feature_names)
    scaled   = scaler.transform(input_df)
    pred     = model.predict(scaled)[0]
    proba    = model.predict_proba(scaled)[0]

    return (
        LABELS[pred],
        f"{proba[pred] * 100:.1f} %",
        f"{proba[0] * 100:.1f} %",
        f"{proba[1] * 100:.1f} %",
    )

SLIDER_CONFIG = {
    "acceleration_x": dict(minimum=-20.0, maximum=20.0, step=0.01, value=0.30),
    "acceleration_y": dict(minimum=-20.0, maximum=20.0, step=0.01, value=9.80),
    "acceleration_z": dict(minimum=-20.0, maximum=20.0, step=0.01, value=0.10),
    "gyro_x":         dict(minimum=-10.0, maximum=10.0, step=0.01, value=0.00),
    "gyro_y":         dict(minimum=-10.0, maximum=10.0, step=0.01, value=0.00),
    "gyro_z":         dict(minimum=-10.0, maximum=10.0, step=0.01, value=0.00),
    "wrist":          dict(minimum=0,     maximum=1,    step=1,    value=0),
}

inputs = [
    gr.Slider(label=feat, **SLIDER_CONFIG.get(feat, dict(minimum=-20.0, maximum=20.0, step=0.01, value=0.0)))
    for feat in feature_names
]

outputs = [
    gr.Textbox(label="Predicted Activity"),
    gr.Textbox(label="Confidence"),
    gr.Textbox(label="P(Walking)"),
    gr.Textbox(label="P(Running)"),
]

demo = gr.Interface(
    fn=predict_activity,
    inputs=inputs,
    outputs=outputs,
    examples=[
        [0.30,  9.81,  0.10, 0.00, 0.01,  0.00, 0],
        [1.20, 14.50, -1.30, 0.80, 0.50, -0.30, 1],
    ],
    title="🏃 Walk vs Run Classifier",
    description="Enter wearable sensor readings to classify activity.\n\n**Model:** Random Forest | **Accuracy:** ~99.2%",
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())   # ← theme moved here in Gradio 6.0