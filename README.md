# CPU Inference After Neural Network Quantization

## Overview

This learning project investigates whether converting a small image-classification neural network from FP32 to INT8 improves inference latency, throughput, and model size on a standard CPU, and measures any resulting accuracy loss.

The project currently focuses on building and understanding the FP32 training pipeline before introducing quantization and CPU benchmarks.

## Research question

How does INT8 quantization affect the accuracy, model size, batch-1 latency, and batch-32 throughput of a small CIFAR-10 CNN compared with its FP32 baseline on the same CPU?

## Current progress

The current `src/train.py` is an intentionally verbose learning script. It now demonstrates:

- Downloading and loading CIFAR-10 with PyTorch.
- Inspecting image tensors, labels, RGB values, and class names.
- Building a small CNN layer by layer.
- Following tensor shapes through convolution, ReLU, max pooling, flattening, and fully connected layers.
- Computing cross-entropy loss and gradients.
- Updating weights with SGD.
- Training for two complete epochs.
- Calculating accuracy across the 50,000-image training set with gradients disabled.

Measured test accuracy and CPU performance results are not available yet.

## Current model

| Stage | Operation | Output per image |
| --- | --- | --- |
| Input | CIFAR-10 RGB image | `3 x 32 x 32` |
| Feature extraction 1 | `Conv2d(3, 8, 3, padding=1)` + ReLU + `MaxPool2d(2, 2)` | `8 x 16 x 16` |
| Feature extraction 2 | `Conv2d(8, 16, 3, padding=1)` + ReLU + `MaxPool2d(2, 2)` | `16 x 8 x 8` |
| Flatten | Flatten feature maps | `1024` |
| Classifier | `Linear(1024, 64)` + ReLU | `64` |
| Output | `Linear(64, 10)` | `10` class scores |

Training currently uses cross-entropy loss, SGD with a learning rate of `0.01`, a batch size of `4`, and two epochs.

## Setup

The recorded development environment uses Windows 11, Python 3.14.6, CPU-only PyTorch 2.13.0, and Torchvision 0.28.0. See `hardware.md` for full hardware details.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the current training script

```powershell
python src/train.py
```

On its first run, the script downloads CIFAR-10 into the ignored `data/` directory. It also opens a Matplotlib image window before training continues.

## Repository structure

```text
.
|-- src/
|   |-- train.py       # Current educational CNN and training flow
|   |-- model.py       # Planned reusable model definition
|   |-- evaluate.py    # Planned test-set evaluation
|   |-- quantize.py    # Planned INT8 quantization
|   `-- benchmark.py   # Planned CPU benchmarks
|-- configs/           # Planned experiment configurations
|-- notebooks/         # Exploratory work
|-- results/           # Generated measurements and figures
|-- reports/           # Technical reports
|-- daily-log.md       # Research progress log
|-- hardware.md        # Recorded experimental environment
`-- requirements.txt   # Python environment snapshot
```

## Roadmap

- Refactor the layer-by-layer script into a reusable `nn.Module`.
- Add held-out CIFAR-10 test-set evaluation.
- Save and measure the trained FP32 model.
- Establish FP32 accuracy, model-size, batch-1 latency, and batch-32 throughput baselines.
- Apply INT8 quantization.
- Repeat the measurements and compare FP32 with INT8.

## Experiment metrics

- Test accuracy
- Model size
- Batch-1 inference latency
- Batch-32 inference throughput
