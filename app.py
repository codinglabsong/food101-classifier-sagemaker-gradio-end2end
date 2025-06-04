import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import gradio as gr

# Load config and class names
cfg = yaml.safe_load(open("config/prod.yaml"))
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

# Build and load model


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(path: str, num_classes: int) -> nn.Module:
    model = build_model(num_classes)
    state = torch.load(path, map_location="cpu")
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model("output/model.pth", len(class_names))

# Preprocessing (must match training)
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(cfg["estimator"]["hyperparameters"]["img-size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)[0]
    return {class_names[i]: float(probs[i]) for i in probs.argsort(descending=True)}


# Example images for the UI
example_dir = "examples"
if os.path.isdir(example_dir):
    examples = [
        [os.path.join(example_dir, f)]
        for f in os.listdir(example_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
else:
    examples = None

# Launch Gradio app
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5, label="Top Classes"),
    title="Food101 Classifier",
    examples=examples,
).launch()
