# CPU Inference After Neural Network Quantization

## Aim

This is a small learning project using a CIFAR-10 CNN to compare FP32 and INT8 inference on a CPU. The aim is to reduce model size and improve inference speed without losing too much accuracy.

## Current progress

The FP32 baseline is set up, and I have started a manual INT8 weight-quantization experiment:

- `SmallCNN` is defined as a reusable PyTorch `nn.Module`.
- `train.py` trains the model for two epochs and saves its weights.
- `evaluate.py` loads the checkpoint and records test accuracy, model size and parameter count.
- `quantize.py` checks an activation range over 100 calibration batches.
- Each convolution and linear layer has its own symmetric weight scale.
- The raw INT8 weight tensors use one quarter of the FP32 weight storage (`4.0x` compression).

This is not an end-to-end INT8 inference model yet. The quantized tensors are not used for prediction, and CPU latency has not been benchmarked.

## FP32 baseline

| Metric | Result |
| --- | ---: |
| CIFAR-10 test accuracy | `54.1%` |
| Saved model size | `0.261 MB` |
| Number of parameters | `67,642` |

The full values are stored in `results/fp32_metrics.json`.

## Model

The model is deliberately small so that it is easy to understand and run on a CPU:

```text
3 x 32 x 32 image
-> Conv(3, 8) -> ReLU -> MaxPool
-> Conv(8, 16) -> ReLU -> MaxPool
-> Flatten
-> Linear(1024, 64) -> ReLU
-> Linear(64, 10)
```

Training currently uses a batch size of `4`, cross-entropy loss, SGD and a learning rate of `0.01`.

## Setup

The recorded environment uses Windows 11, Python 3.14.6 and CPU-only PyTorch 2.13.0.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force checkpoints, results
```

## Run

Run the scripts in this order:

```powershell
python src/train.py
python src/evaluate.py
python src/quantize.py
```

The first command trains and saves the model, the second records the FP32 baseline, and the third runs the manual weight-quantization experiment. The training and evaluation scripts also display an example image with Matplotlib.

## Main files

- `src/model.py` - the `SmallCNN` model.
- `src/train.py` - data loading, training, accuracy checks and checkpoint saving.
- `src/evaluate.py` - test-set evaluation and FP32 metrics.
- `src/quantize.py` - activation-range checks and manual INT8 weight quantization.
- `src/benchmark.py` - planned CPU benchmark work.

## Next steps

- Use quantized weights and activations during inference.
- Measure INT8 test accuracy.
- Measure batch-1 latency and batch-32 throughput.
- Compare FP32 and INT8 accuracy, size and speed.
