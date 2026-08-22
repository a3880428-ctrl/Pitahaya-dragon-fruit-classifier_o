"""
Dragon Fruit (Pitahaya) Stem Disease Classifier — Gradio app.

Loads 'dragon_fruit_classifier.pt', the REAL checkpoint from
train_dragon_fruit_classifier.py (94.4% val accuracy, verified architecture,
verified data source). No placeholder heads, no untrained-weights fallback
disguising itself as a real result.
"""

import os
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models, transforms

# NOTE: nest_asyncio deliberately NOT used here — not needed outside Colab,
# and it conflicts with newer Python/uvicorn's asyncio.run() signature anyway.


# ==========================================
# LOAD MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running inference on device: {device}")

CHECKPOINT_PATH = "dragon_fruit_classifier.pt"

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"'{CHECKPOINT_PATH}' not found. Run train_dragon_fruit_classifier.py "
        "in this same Colab session first — this app has no fallback to "
        "untrained weights on purpose, since that's what caused every "
        "confusing result earlier in this project."
    )

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
class_names = checkpoint['class_names']
num_classes = len(class_names)
val_acc = checkpoint.get('val_acc', None)

model = models.resnet18(weights=None)  # weights come from checkpoint, not ImageNet, at load time
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(device)
model.eval()

print(f"[*] Loaded checkpoint. Classes: {class_names}")
if val_acc is not None:
    print(f"[*] Reported validation accuracy at save time: {val_acc:.4f}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ==========================================
# INFERENCE
# ==========================================
def predict_disease(image):
    if image is None:
        return "No image provided", {}

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    # Full probability breakdown for gr.Label
    prob_dict = {class_names[i]: float(probs[i]) for i in range(num_classes)}

    top_line = f"{pred_label} ({confidence * 100:.1f}% confidence)"
    return top_line, prob_dict


# ==========================================
# GRADIO INTERFACE
# ==========================================
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Dragon Fruit (Pitahaya) Stem Photo"),
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Label(label="Full Probability Breakdown", num_top_classes=6),
    ],
    title="Dragon Fruit Stem Disease Classifier",
    description=(
        f"6-class disease classifier (Anthracnose, Brown Stem Spot, Gray Blight, "
        f"Healthy, Soft Rot, Stem Canker). Trained on 724 real images "
        f"(Mendeley 'Dragon Fruit Stem Disease' dataset)."
        + (f" Validation accuracy at save time: {val_acc*100:.1f}%." if val_acc else "")
        + "\n\nNote: Anthracnose and Stem Canker are the two classes most often "
        "confused with each other in validation — treat predictions between "
        "those two with extra caution."
    ),
)

if __name__ == "__main__":
    # Render (and most free hosts) assign the port dynamically via $PORT,
    # and require binding to 0.0.0.0, not localhost/Colab's tunnel.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, debug=False)
