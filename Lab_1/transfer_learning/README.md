# Transfer Learning: image classification

This part of Lab 1 compares two Transfer Learning strategies on three MNIST
classes: `digit_0`, `digit_1`, and `digit_2`. MNIST is a public handwritten
digit-image dataset; the experiment uses a balanced three-class subset. The experiment uses ImageNet
pretrained MobileNetV2. It is a deliberately compact architecture (about 3.5M
parameters), so it is practical on a CPU while retaining useful pretrained
visual features.

## Reproduction

From the repository root, create an environment, install the dependencies, and
run the experiment:

```powershell
python -m pip install -r requirements.txt
python Lab_1/transfer_learning/train_transfer_learning.py --data-dir Lab_1/transfer_learning/data --output-dir Lab_1/transfer_learning/artifacts
```

The script downloads MNIST and the official ImageNet weights on its first
run. It creates a balanced training set (600 images per class) and test set
(150 images per class), fixes the random seed to 42, and writes:

- `metrics.csv` with Accuracy, macro Precision, macro Recall, and macro F1;
- one confusion matrix for each strategy;
- an experiment configuration file and model checkpoints.

Feature extraction freezes all MobileNetV2 feature layers and trains only a new
three-class classifier. Fine-tuning unfreezes the final two feature modules and
trains them with a lower learning rate (`1e-4` versus `1e-3`).
