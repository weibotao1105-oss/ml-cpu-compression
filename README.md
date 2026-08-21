# CPU Inference After Neural Network Quantization

## Aim

This is a small learning project about running neural networks on a CPU. I am using a CNN trained on CIFAR-10 to compare a normal FP32 model with an INT8 quantized model.

The main question is whether INT8 can reduce model size and improve inference speed without losing too much accuracy.

## Current progress

The FP32 part of the project is now mostly set up:

- `SmallCNN` is defined as a reusable PyTorch `nn.Module`.
- `train.py` trains the model for two epochs and saves its weights.
- Training and test accuracy are calculated separately.
- `evaluate.py` loads the saved weights and evaluates the test set.
- The evaluation records test accuracy, model size and parameter count.
- FP32 results can be saved to `results/fp32_metrics.json`.

INT8 quantization and CPU latency benchmarks have not been added yet.

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

Train and save the FP32 model first:

```powershell
python src/train.py
```

Then load the checkpoint and evaluate it:

```powershell
python src/evaluate.py
```

Both scripts display an example CIFAR-10 image with Matplotlib.

## Main files

- `src/model.py` - the `SmallCNN` model.
- `src/train.py` - data loading, training, accuracy checks and checkpoint saving.
- `src/evaluate.py` - test-set evaluation and FP32 metrics.
- `src/quantize.py` - planned INT8 quantization work.
- `src/benchmark.py` - planned CPU benchmark work.
- `hardware.md` - the machine used for the experiment.
- `daily-log.md` - short notes on project progress.

## Next steps

- Quantize the FP32 model to INT8.
- Measure batch-1 latency and batch-32 throughput.
- Compare FP32 and INT8 accuracy, size and speed.
