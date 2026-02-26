# Network Configuration Document — DL Competition 01

## Architecture Description

**Model: SceneCNN** — A convolutional neural network with 3 convolutional blocks followed by a fully connected classifier.

### Convolutional Feature Extractor

| Block | Layer             | Details                                |
|-------|-------------------|----------------------------------------|
| 1     | Conv2d            | 1 → 32 filters, 3×3 kernel, padding=1 |
|       | BatchNorm2d(32)   |                                        |
|       | ReLU              |                                        |
|       | MaxPool2d(2×2)    | Output: 32 × 75 × 75                  |
| 2     | Conv2d            | 32 → 64 filters, 3×3 kernel, padding=1|
|       | BatchNorm2d(64)   |                                        |
|       | ReLU              |                                        |
|       | MaxPool2d(2×2)    | Output: 64 × 37 × 37                  |
| 3     | Conv2d            | 64 → 128 filters, 3×3 kernel, padding=1|
|       | BatchNorm2d(128)  |                                        |
|       | ReLU              |                                        |
|       | MaxPool2d(2×2)    | Output: 128 × 18 × 18                 |

### Classifier Head

| Layer          | Details                  |
|----------------|--------------------------|
| Flatten        | 128 × 18 × 18 = 41,472  |
| Linear         | 41,472 → 256            |
| ReLU           |                          |
| Dropout(0.5)   |                          |
| Linear         | 256 → 6 (output classes) |

## Input Size and Preprocessing

- **Input size:** 1 × 150 × 150 (grayscale)
- **Training augmentation:** RandomHorizontalFlip(p=0.5), RandomRotation(15°)
- **Normalization:** mean=0.5, std=0.5

## Loss Function

- **CrossEntropyLoss** — standard for multi-class classification; combines LogSoftmax and NLLLoss.

## Optimizer and Hyperparameters

| Parameter       | Value         |
|-----------------|---------------|
| Optimizer       | Adam          |
| Learning rate   | 0.001         |
| Weight decay    | 1e-4          |
| Batch size      | 64            |
| Epochs          | 25            |
| Scheduler       | ReduceLROnPlateau (factor=0.5, patience=3) |

## Regularization Methods

1. **Dropout (p=0.5):** Applied in the classifier head to prevent overfitting.
2. **Batch Normalization:** After each convolutional layer for training stability.
3. **Data Augmentation:** Random horizontal flip and rotation during training.
4. **Weight Decay (L2 regularization):** 1e-4 applied via optimizer.
5. **Learning Rate Scheduling:** Reduces LR by 50% when validation loss stalls for 3 epochs.

## Model Selection Strategy

- The model is evaluated on the validation set after every epoch.
- The model weights with the **highest validation accuracy** are saved (`best_model.pth`).
- These best weights are loaded before generating competition predictions.
- A final sanity check is run on the held-out test set (separate from validation) to estimate generalization.
