"""
Dragon Fruit (Pitahaya) Stem Disease Classifier — ONNX Runtime version.

Deliberately does NOT import torch or torchvision. Those libraries, even the
CPU-only build, were exceeding Render's free-tier 512MB RAM limit at runtime
(confirmed: two separate OOM crashes with torch/torchvision loaded). This
version uses onnxruntime instead — a much lighter inference-only runtime —
verified to produce numerically identical predictions to the original
PyTorch model (max output difference: 0.000001 for fp32 ONNX; same
predicted class after int8 quantization).
"""

import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import gradio as gr

MODEL_PATH = "dragon_fruit_classifier_quant.onnx"
CLASS_NAMES_PATH = "class_names.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' not found. Make sure it's uploaded to this repo "
        "alongside this script — no fallback to untrained weights, on purpose."
    )
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"'{CLASS_NAMES_PATH}' not found.")

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

session = ort.InferenceSession(MODEL_PATH)
print(f"[*] Loaded ONNX model. Classes: {class_names}")

# ImageNet normalization stats (same ones used during training/export)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0       # [H, W, 3], 0-1
    arr = (arr - MEAN) / STD                                 # normalize
    arr = arr.transpose(2, 0, 1)                              # -> [3, H, W]
    arr = np.expand_dims(arr, axis=0).astype(np.float32)       # -> [1, 3, H, W]
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_disease(image):
    if image is None:
        return "No image provided", {}

    input_tensor = preprocess(image)
    logits = session.run(None, {"input": input_tensor})[0][0]  # [num_classes]
    probs = softmax(logits)

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    top_line = f"{pred_label} ({confidence * 100:.1f}% confidence)"
    return top_line, prob_dict


demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Dragon Fruit (Pitahaya) Stem Photo"),
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Label(label="Full Probability Breakdown", num_top_classes=6),
    ],
    title="Dragon Fruit Stem Disease Classifier",
    description=(
        "6-class disease classifier (Anthracnose, Brown Stem Spot, Gray Blight, "
        "Healthy, Soft Rot, Stem Canker). Trained on 724 real images "
        "(Mendeley 'Dragon Fruit Stem Disease' dataset), 94.4% validation accuracy. "
        "Running on ONNX Runtime for low-memory hosting.\n\n"
        "Note: Anthracnose and Stem Canker are the two classes most often "
        "confused with each other in validation — treat predictions between "
        "those two with extra caution."
    ),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, debug=False)
