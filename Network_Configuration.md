# Network Configuration Document — DL Competition 01

## Architecture Description

**Model: DeepResidualMLP** — A deep Multi-Layer Perceptron with residual (skip) connections for 6-class image classification. No convolutional layers are used.

### Architecture Overview

```
Input: 3 × 150 × 150 = 67,500 (flattened RGB image)
  → Stem:       Linear(67,500 → 2,048) + BN + GELU + Dropout(0.3)
  → ResBlock 1: 2,048 → 2,048  (identity skip connection)
  → ResBlock 2: 2,048 → 1,024  (projection skip connection)
  → ResBlock 3: 1,024 → 512    (projection skip connection)
  → ResBlock 4: 512   → 256    (projection skip connection)
  → Head:       Linear(256 → 128) + BN + GELU + Dropout(0.2) + Linear(128 → 6)
```

### Detailed Layer Breakdown

#### Stem (Input Projection)

| Layer          | Details                                      |
|----------------|----------------------------------------------|
| Flatten        | 3 × 150 × 150 = 67,500 input features       |
| Linear         | 67,500 → 2,048                               |
| BatchNorm1d    | 2,048                                        |
| GELU           |                                              |
| Dropout(0.3)   |                                              |

#### Residual Block Structure (×4)

Each residual block follows this pattern:

| Component      | Details                                      |
|----------------|----------------------------------------------|
| Linear 1       | in_dim → out_dim                             |
| BatchNorm1d    | out_dim                                      |
| GELU           |                                              |
| Dropout        | 0.4 (blocks 1-3) or 0.3 (block 4)           |
| Linear 2       | out_dim → out_dim                            |
| BatchNorm1d    | out_dim                                      |
| + Skip         | Identity (same dim) or Linear projection     |
| GELU           |                                              |
| Dropout        | 0.4 (blocks 1-3) or 0.3 (block 4)           |

#### Residual Blocks — Dimension Progression

| Block | Input Dim | Output Dim | Skip Connection          | Dropout |
|-------|-----------|------------|--------------------------|---------|
| 1     | 2,048     | 2,048      | Identity                 | 0.4     |
| 2     | 2,048     | 1,024      | Linear(2048→1024) + BN   | 0.4     |
| 3     | 1,024     | 512        | Linear(1024→512) + BN    | 0.4     |
| 4     | 512       | 256        | Linear(512→256) + BN     | 0.3     |

#### Classification Head

| Layer          | Details                                      |
|----------------|----------------------------------------------|
| Linear         | 256 → 128                                    |
| BatchNorm1d    | 128                                          |
| GELU           |                                              |
| Dropout(0.2)   |                                              |
| Output Linear  | 128 → 6 (output classes)                     |

### Total Parameters

Approximately **148M** parameters. The stem layer dominates (67,500 × 2,048 ≈ 138M), which is inherent to MLP architectures operating on flattened images.

### Weight Initialization

- **Linear layers:** Kaiming Normal (He initialization), biases set to zero
- **BatchNorm layers:** weight = 1, bias = 0

## Input Size and Preprocessing

- **Input size:** 3 × 150 × 150 (RGB), flattened to 67,500-dimensional vector
- **Training augmentation (aggressive, to compensate for lack of spatial inductive bias):**
  - Resize to 160×160 + RandomCrop to 150×150 (translation invariance)
  - RandomHorizontalFlip (p=0.5)
  - RandomRotation (20°)
  - RandomAffine (translate=10%, scale=85%-115%)
  - ColorJitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
  - RandomErasing (p=0.2, scale=2%-15%) — applied after normalization
- **Evaluation preprocessing:** Resize to 150×150 only (no augmentation)
- **Normalization:** ImageNet statistics — mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Loss Function

- **CrossEntropyLoss** with **label smoothing = 0.1** — softens one-hot targets to reduce overconfidence and improve generalization. Targets become 0.9 for the correct class and 0.1/(K-1) for others.

## Optimizer and Hyperparameters

| Parameter       | Value                                      |
|-----------------|--------------------------------------------|
| Optimizer       | AdamW (decoupled weight decay)             |
| Learning rate   | 0.001                                      |
| Weight decay    | 5e-4                                       |
| Batch size      | 64                                         |
| Max epochs      | 60                                         |
| Scheduler       | CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-6) |
| Gradient clip   | max_norm = 1.0                             |

### Scheduler Details

CosineAnnealingWarmRestarts uses cyclic cosine decay with warm restarts:
- First cycle: 10 epochs
- Second cycle: 20 epochs (T_mult=2)
- Third cycle: 40 epochs
- Minimum LR: 1e-6

This helps escape local minima and explore different regions of the loss landscape.

## Regularization Methods

1. **Residual Connections:** Skip connections in every block enable deeper training and mitigate vanishing gradients. Projection shortcuts (Linear + BN) are used when dimensions change.

2. **Dropout (p=0.2 to 0.4):** Applied throughout the network with decreasing rates in deeper layers:
   - Stem: 0.3
   - Residual blocks 1-3: 0.4
   - Residual block 4: 0.3
   - Classification head: 0.2

3. **Batch Normalization (BatchNorm1d):** After every linear layer, including inside skip connections, for training stability and implicit regularization.

4. **Weight Decay (L2 Regularization):** 5e-4 via AdamW's decoupled weight decay, which applies decay directly to weights rather than through the gradient (more principled than L2 in Adam).

5. **Label Smoothing (0.1):** Prevents the model from becoming overconfident on training examples, leading to better-calibrated predictions.

6. **Mixup (α=0.2):** During training, random pairs of images and labels are linearly interpolated using a Beta-distributed mixing coefficient. This creates "virtual" training examples between classes, smoothing decision boundaries and acting as a strong regularizer.

7. **Data Augmentation (aggressive):** RandomCrop, horizontal flip, rotation, affine transforms, color jitter, and random erasing. More aggressive than typical CNN pipelines because MLPs lack built-in translation invariance.

8. **Random Erasing (p=0.2):** Randomly masks rectangular patches in the image tensor, forcing the model to rely on multiple regions rather than a single discriminative area.

9. **Gradient Clipping (max_norm=1.0):** Prevents gradient explosions, especially important in the early layers where the large input dimension can cause instability.

10. **Early Stopping (patience=10):** Training stops if validation accuracy does not improve for 10 consecutive epochs, preventing overfitting.

## Design Rationale — MLP Approach

The MLP approach flattens the entire image into a vector, treating each pixel as an independent input feature. This introduces several challenges and design decisions:

- **No spatial hierarchy:** Unlike CNNs, the MLP does not inherently learn local patterns (edges, textures) through kernels and pooling. All spatial relationships must be learned from data, which requires more training data and stronger regularization.

- **Large parameter count:** The stem layer alone has ~138M parameters (67,500 × 2,048). This is unavoidable when projecting high-dimensional flattened images and makes regularization critical.

- **RGB over Grayscale:** Color information (3 channels) is preserved because it provides strong discriminative signal for scene classification (green forests, blue sea, gray buildings). This triples the input dimension but significantly improves accuracy.

- **Residual connections:** Without skip connections, training deep MLPs on image data often leads to degradation (deeper ≠ better). Residual blocks solve this by allowing gradient flow through shortcuts.

- **GELU over ReLU:** The Gaussian Error Linear Unit provides smoother gradients than ReLU, which helps in the early layers where many neurons could "die" with ReLU due to the high-dimensional input.

- **Mixup + Label Smoothing:** These two techniques work synergistically — Mixup creates soft training examples while label smoothing prevents overconfident predictions, together producing smoother and more generalizable decision boundaries.

## Model Selection Strategy

- The model is evaluated on the **full validation set (~2,400 samples)** after every epoch, obtained via stratified 80/20 split of seg_test to maintain class balance.
- The model weights with the **highest validation accuracy** are saved (`best_model.pth`) using `copy.deepcopy` of the state dict.
- **Early stopping** with patience=10 halts training if no improvement is observed.
- These best weights are loaded before generating competition predictions on comp_test.
- A final sanity check is run on the held-out test set (~600 samples, separate from validation) to estimate generalization performance.
- Per-class accuracy is computed on the test set to identify potential weaknesses across the 6 scene classes.
