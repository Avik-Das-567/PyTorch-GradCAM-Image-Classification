# Deep Learning with PyTorch: Grad-CAM Visual Explainability

This project builds a convolutional neural network in PyTorch for three-class image classification and applies Gradient-weighted Class Activation Mapping (Grad-CAM) to explain the model's class-specific predictions. The classifier distinguishes between cucumber, eggplant, and mushroom images, then uses gradients flowing into the final convolutional representation to produce heatmaps that highlight the image regions most responsible for a selected class score.

Grad-CAM is useful because it connects classification output with spatial evidence. Instead of only returning logits or predicted labels, the model can be inspected visually to understand whether its decision is driven by meaningful object regions or by irrelevant background features.

## Objectives

- Understand the end-to-end Grad-CAM pipeline for image classification.
- Build a custom image dataset workflow from CSV image paths and numeric labels.
- Train and validate a compact convolutional neural network using PyTorch.
- Capture class-specific gradients from the final convolutional feature maps.
- Generate and visualize localization heatmaps over the original validation images.

## Dataset

The project uses a small supervised image dataset described by a CSV file with two columns:

- `img_path`: relative path to the image file.
- `label`: integer-encoded class label.

The class mapping used throughout the notebook is:

| Label | Class |
| --- | --- |
| `0` | Cucumber |
| `1` | Eggplant |
| `2` | Mushroom |

The dataset is split into training and validation subsets using an 80/20 split with `random_state = 42`. After splitting, the notebook reports:

- Training examples: `148`
- Validation examples: `38`
- Training batches: `10`
- Validation batches: `3`
- Batch image tensor shape: `torch.Size([16, 3, 227, 227])`
- Batch label tensor shape: `torch.Size([16])`

Images are represented as RGB tensors with channel-first layout, which is the expected input format for PyTorch convolutional layers.

## Preprocessing and Augmentation

The preprocessing pipeline is implemented with Albumentations. Training data uses geometric augmentation plus normalization, while validation data uses deterministic normalization only.

Training augmentations:

- `A.Rotate()`
- `A.HorizontalFlip(p = 0.5)`
- `A.VerticalFlip(p = 0.5)`
- `A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])`

Validation preprocessing:

- `A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])`

The normalization statistics match the common ImageNet RGB mean and standard deviation values. For visualization, normalized tensors are denormalized before plotting so Grad-CAM overlays are rendered on interpretable RGB images.

## Model Architecture

The project defines a custom CNN named `ImageModel`. It separates convolutional feature extraction from the dense classification head and includes a gradient hook on the final convolutional activations so Grad-CAM can access backward-pass gradients.

### Feature Extractor

The feature extractor contains four convolutional stages:

| Stage | Layer | Output Channels | Kernel | Padding | Activation / Pooling |
| --- | --- | ---: | --- | --- | --- |
| 1 | `Conv2d(3, 16)` | 16 | `5 x 5` | `1` | `ReLU`, `MaxPool2d(4, stride=2)` |
| 2 | `Conv2d(16, 16)` | 16 | `5 x 5` | `1` | `ReLU`, `MaxPool2d(4, stride=2)` |
| 3 | `Conv2d(16, 32)` | 32 | `5 x 5` | `1` | `ReLU`, `MaxPool2d(4, stride=2)` |
| 4 | `Conv2d(32, 64)` | 64 | `5 x 5` | `1` | `ReLU` |

After the feature extractor, the model applies one additional `MaxPool2d(kernel_size = 4, stride = 2)` before passing the tensor into the classifier.

### Classifier

The classifier maps the pooled convolutional representation to three class logits:

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(6400, 2048),
    nn.ReLU(),
    nn.Linear(2048, 3)
)
```

The output dimension of `3` corresponds directly to cucumber, eggplant, and mushroom.

### Gradient Hook for Explainability

During the forward pass, the output of the final convolutional feature extractor is registered with:

```python
h = x.register_hook(self.activations_hook)
```

The hook stores the gradient tensor during backpropagation. This makes the model explainability-ready without changing the classifier output. The model also exposes two helper methods:

- `get_activation_gradients()`: returns gradients captured from the final convolutional activations.
- `get_activation(x)`: returns the feature extractor activations for a given input image.

## Training Configuration

The training setup uses a standard supervised multi-class classification objective:

| Component | Value |
| --- | --- |
| Device | `cuda` |
| Batch size | `16` |
| Learning rate | `0.001` |
| Epochs | `20` |
| Optimizer | `torch.optim.Adam` |
| Loss function | `torch.nn.CrossEntropyLoss` |
| Checkpoint file | `best_weights.pt` |

The training function runs the model in training mode, moves each image and label batch to the configured device, clears optimizer gradients, computes logits, evaluates cross-entropy loss, performs backpropagation, and updates model parameters.

The evaluation function switches the model to evaluation mode and computes average validation loss across validation batches. The notebook tracks the best validation loss and saves model weights whenever validation loss improves.

## Project Workflow

1. Clone and load the image classification dataset.
2. Import the core libraries: PyTorch, Torchvision, Pandas, NumPy, OpenCV, Matplotlib, Albumentations, Scikit-learn, and helper utilities.
3. Configure dataset paths, device, batch size, learning rate, and epoch count.
4. Read the training CSV and inspect the image path and label structure.
5. Split the dataset into training and validation subsets with an 80/20 split.
6. Define training and validation transforms.
7. Build custom dataset objects from the split dataframes.
8. Create PyTorch `DataLoader` objects for mini-batch training and validation.
9. Define the custom CNN architecture with activation-gradient capture.
10. Train the model for 20 epochs using Adam and cross-entropy loss.
11. Save the best-performing model weights based on validation loss.
12. Generate Grad-CAM heatmaps for selected validation images and target class logits.
13. Overlay the heatmaps on denormalized RGB images to interpret class-specific localization.

## Grad-CAM Method

The Grad-CAM function receives a trained model, a single input image, a selected class score, and the desired heatmap size. It computes the localization map with the following sequence:

1. Backpropagate from the selected class logit using `label.backward()`.
2. Retrieve gradients captured from the final convolutional activation tensor.
3. Average the gradients across spatial dimensions with `torch.mean(gradients, dim = [0, 2, 3])`.
4. Retrieve the corresponding forward activations from the feature extractor.
5. Weight each activation channel by its pooled gradient importance.
6. Average the weighted activation channels into a single coarse localization map.
7. Apply ReLU to keep only positive evidence for the selected class.
8. Normalize the heatmap by its maximum value.
9. Resize the heatmap to `227 x 227` with OpenCV.

This produces a class-discriminative heatmap where higher intensity regions represent stronger positive contribution to the selected output class.

## Results

The model was trained for 20 epochs. Validation loss improved several times during training, triggering checkpoint saves when a new best validation loss was observed.

| Epoch | Train Loss | Validation Loss | Checkpoint |
| ---: | ---: | ---: | --- |
| 1 | `0.9618587732` | `0.8046830098` | Saved |
| 2 | `0.6551892877` | `0.4199364583` | Saved |
| 5 | `0.2429185130` | `0.1747766553` | Saved |
| 8 | `0.1536747145` | `0.0932590229` | Saved |
| 14 | `0.0871614659` | `0.0813356834` | Saved |
| 17 | `0.1165483233` | `0.0609933893` | Saved |
| 20 | `0.0487792769` | `0.0863785949` | Final epoch |

The best validation loss was `0.06099338928470388` at epoch 17. The final epoch ended with a training loss of `0.04877927687484771` and a validation loss of `0.08637859486043453`.

Qualitative Grad-CAM results were produced on multiple validation samples. The notebook evaluates selected images from the validation set, computes class logits, selects a class-specific score such as `pred[0][0]`, `pred[0][1]`, or `pred[0][2]`, and backpropagates from that score to create the heatmap. The plotted outputs contain the original denormalized image, the activation heatmap, and the heatmap overlay. These visualizations make it possible to compare how the same input image activates different class explanations and whether the highlighted regions align with the visible object.

## Technical Stack

- Python
- PyTorch
- Torchvision
- Albumentations
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- tqdm

## Key Takeaways

This project demonstrates a complete explainable image classification workflow in PyTorch. It combines a compact CNN classifier with gradient-based visual attribution so that predictions can be inspected spatially. The final convolutional activations provide the visual feature maps, the backward gradients provide class-specific importance weights, and the resulting Grad-CAM overlays show which image regions support each selected class prediction.
