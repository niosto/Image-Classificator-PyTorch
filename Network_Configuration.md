# Network Configuration Document — DL Competition 01

## Architecture Description

**Model: SceneMLP** — A Multi-Layer Perceptron (fully connected neural network) with 3 hidden layers for image classification.

### MLP Architecture

| Layer          | Details                                      |
|----------------|----------------------------------------------|
| Flatten        | 1 × 150 × 150 = 22,500 input features       |
| Linear 1       | 22,500 → 1,024                               |
| BatchNorm1d    | 1,024                                        |
| ReLU           |                                              |
| Dropout(0.4)   |                                              |
| Linear 2       | 1,024 → 512                                  |
| BatchNorm1d    | 512                                          |
| ReLU           |                                              |
| Dropout(0.4)   |                                              |
| Linear 3       | 512 → 256                                    |
| BatchNorm1d    | 256                                          |
| ReLU           |                                              |
| Dropout(0.3)   |                                              |
| Output Linear  | 256 → 6 (output classes)                     |

### Total Parameters

Approximately **23.6M** parameters (dominated by the first linear layer: 22,500 × 1,024 = ~23M).

## Input Size and Preprocessing

- **Input size:** 1 × 150 × 150 (grayscale), flattened to 22,500-dimensional vector
- **Training augmentation:**
  - RandomHorizontalFlip(p=0.5)
  - RandomVerticalFlip(p=0.2)
  - RandomRotation(20°)
  - RandomAffine(translate=10%, scale=90%-110%)
  - ColorJitter(brightness=0.3, contrast=0.3)
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
| Epochs          | 30            |
| Scheduler       | ReduceLROnPlateau (factor=0.5, patience=3) |

## Regularization Methods

1. **Dropout (p=0.4 and p=0.3):** Applied after each hidden layer to prevent overfitting. Higher dropout in earlier layers (more parameters), lower in deeper layers.
2. **Batch Normalization (BatchNorm1d):** After each linear layer for training stability and faster convergence.
3. **Data Augmentation (enhanced):** Horizontal/vertical flip, rotation, affine transforms, and brightness/contrast jitter during training. More aggressive augmentation is critical for MLPs since they lack spatial inductive bias.
4. **Weight Decay (L2 regularization):** 1e-4 applied via optimizer.
5. **Learning Rate Scheduling:** Reduces LR by 50% when validation loss stalls for 3 epochs.

## Design Rationale — MLP vs CNN

The MLP approach flattens the entire image into a vector, treating each pixel as an independent feature. This means:
- **No spatial hierarchy:** Unlike CNNs, the MLP does not inherently learn local patterns (edges, textures). It must learn all spatial relationships from scratch.
- **More parameters:** The first layer alone has ~23M parameters vs a CNN's first conv layer with ~320.
- **Stronger augmentation needed:** To compensate for the lack of translation invariance (which CNNs get from convolutions + pooling), we apply more aggressive data augmentation.
- **More regularization:** Higher dropout rates and batch normalization are essential to prevent overfitting given the large parameter count.

## Model Selection Strategy

- The model is evaluated on the validation set after every epoch.
- The model weights with the **highest validation accuracy** are saved (`best_model.pth`).
- These best weights are loaded before generating competition predictions.
- A final sanity check is run on the held-out test set (separate from validation) to estimate generalization.