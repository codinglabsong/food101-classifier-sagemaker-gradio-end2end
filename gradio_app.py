from collections import OrderedDict
import torch, yaml
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import gradio as gr

cfg = yaml.safe_load(open("config/prod.yaml"))

# 1. Recreate model class
def build_model(num_classes):
    """
    Builds an EfficientNet-B2 model with a custom classification head.

    Args:
        num_classes (int): Number of output classes for the classification head.

    Returns:
        nn.Module: The modified EfficientNet-B2 model.
    """
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# 2. Load class names
# Load class names from file
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

# 3. Build and load the model
num_classes = len(class_names)
model = build_model(num_classes)

# If you see _orig_mod keys, strip the prefix! (Due to possibilty of saving compiled version of model during training)
ckpt = torch.load("output/model.pth", map_location='cpu')
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    if k.startswith('_orig_mod.'):
        new_state_dict[k[len('_orig_mod.'):]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()

# 4. Preprocessing: same as test transforms in train.py
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(cfg["estimator"]["hyperparameters"]["img-size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], 
                         [0.229,0.224,0.225])
])

# 5. Inference function
def predict(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)                                      # shape: [1, 101]
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()   # shape: [101]
        sorted_indices = probs.argsort()[::-1]                      # descending order
        result = {class_names[i]: float(probs[i]) for i in sorted_indices}
    return result

# 6. Gradio app
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=101, label="Class Probabilities"),
    title="Food101 Classifier"
).launch()