# Model Card: Food101 End-to-End Classifier

This card describes the model produced by the training pipeline in this repository. The model is an EfficientNet-B2 network fine-tuned on the Food‑101 dataset.

## Model Details
- **Architecture**: EfficientNet‑B2 with an ImageNet pre-trained backbone.
- **Framework**: PyTorch
- **Training Data**: [Food‑101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) images resized and normalized as specified in `config/prod.yaml`.
- **Evaluation Metrics**: Top‑1 accuracy on the Food‑101 validation set. Typical scores are around 80%.

## Intended Use
The classifier is meant for educational demos and casual food recognition. It is **not** intended for nutritional, medical, or safety‑critical applications.

## Limitations
- Performance degrades on images of dishes not represented in Food‑101.
- The model may misclassify visually similar foods.
- Only supports images up to 256×256 pixels due to preprocessing.

## Ethical Considerations
See [ETHICS.md](./ETHICS.md) for guidance on data use, privacy, and potential biases.

## Citation
If you use this model in academic work, please cite the Food‑101 dataset paper and mention this repository.

## Contact
For issues or questions, open an issue on the repository.
