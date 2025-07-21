# Ethics Statement

This repository provides code for training and deploying a food image classifier on the Food‑101 dataset. The goal is educational: to show how to build an end‑to‑end computer vision pipeline using PyTorch, Amazon SageMaker, and Gradio.

## Data and Privacy
- **Dataset**: Images come from the public [Food‑101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) dataset and contain typical photos of dishes. No personal or sensitive information is knowingly included.
- **User data**: Inference via the Gradio demo processes uploaded images in memory only. No user data is stored.

## Potential Biases and Limitations
- **Biases**: The dataset covers 101 dish categories, mostly Western cuisine, and may not represent all food cultures. Predictions outside this scope may be unreliable.
- **Model use**: This classifier is not intended for health, medical, or safety‑critical decisions. It should only be used for informal food recognition tasks.

## Responsible Use
- Do not use the model to identify individuals or infer personal attributes from images.
- When sharing models trained with this code, clearly disclose the dataset and limitations.
- Respect copyright and privacy laws when collecting additional data.

## Contact
Please open an issue on this repository if you discover ethical concerns or misuse potential.
