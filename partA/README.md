## README

### Project Overview

This project presents two pipelines for image classification using the **iNaturalist 12K** dataset:

- **`partA.py`**: A custom CNN model with hyperparameter tuning using **Weights & Biases Sweeps**.
- **`partB.py`**: A fine-tuned **ResNet50** backbone using transfer learning, logging performance and evaluation through **WandB**.

---

## Dataset

The dataset structure should follow this format:

```
/kaggle/input/nature-12k/inaturalist_12K/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── val/
        ├── class1/
        ├── class2/
        └── ...
```

## partB.py - Transfer Learning with ResNet50

### How to Use

1. **Pretrained Backbone**:

   - Loads `ResNet50` from torchvision with pretrained ImageNet weights.
   - Freezes all layers except the final classifier head.

2. **Training**:

   - Dataset is split 80:20 for train/val from `train/` folder.
   - Logs training/validation loss & accuracy to WandB.
   - Saves best model as `models/best_model_<run_name>.pth`.

3. **Evaluation**:

   - Evaluates final model on the `val/` folder.
   - Logs class-wise accuracy, confusion matrix, and plots.

4. **Run Entire Pipeline**:
   ```bash
   python partB.py
   ```

---

## Outputs & Artifacts

- 📁 `models/`: Contains saved model weights.
- 📁 `plots/`: Includes training curve plots and confusion matrices.
- 📁 `wandb/`: All training logs and artifacts are pushed to Weights & Biases.

---

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.10
- torchvision
- scikit-learn
- matplotlib
- wandb
- PIL
- numpy

Install dependencies:

```bash
pip install torch torchvision wandb scikit-learn matplotlib
```

---

## Quick Notes

- Set the correct dataset and model path inside the scripts (`/kaggle/input/nature-12k/inaturalist_12K`).
- Ensure your WandB API key is set:
  ```bash
  wandb login
  ```
