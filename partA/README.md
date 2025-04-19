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

---

## partA.py - Custom CNN with WandB Sweeps

### 🔧 How to Use

1. **Sweep Setup and Launch**:

   - Define sweep configurations including learning rate, batch size, activation, filter organization, etc.
   - WandB sweep is initiated in `main()` using:
     ```python
     sweep_id = wandb.sweep(sweep_config, project="A2")
     wandb.agent(sweep_id, train, count=15)
     ```

2. **Train Function**

   - Trains CNN with optional batch norm and dropout.
   - Saves `best_model.pth` and `best_config.json` locally and logs them on WandB.

3. **Test Function**

   - Loads best run from the sweep and evaluates on the test set.
   - Saves a grid visualization of predictions as `test_predictions_grid.png`.

4. **Run Entire Pipeline**:
   ```bash
   python partA.py
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
