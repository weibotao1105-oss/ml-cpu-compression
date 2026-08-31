# ML CPU Compression

This project explores how INT8 post-training quantization affects a small image classifier running on a CPU. I built an FP32 baseline, converted it to a PT2E quantized graph, and compared accuracy and raw weight storage before starting CPU performance measurements.

## Goal

The main comparison is between the original FP32 `SmallCNN` and an INT8 post-training quantized (PTQ) version. I am measuring classification accuracy and weight storage first; CPU latency and throughput benchmarking are still in progress.

## Model and Dataset

The model is trained and evaluated on CIFAR-10. `SmallCNN` has two convolution, ReLU, and max-pooling blocks, followed by a 64-unit fully connected layer and a 10-class output layer. Training uses `ToTensor`, batches of 4, cross-entropy loss, SGD, and two epochs.

## Project Flow

```text
CIFAR-10
-> SmallCNN
-> FP32 training
-> FP32 evaluation and checkpoint
-> torch.export
-> prepare_pt2e with X86InductorQuantizer
-> calibration
-> convert_pt2e
-> INT8 quantized graph
-> torch.compile
-> CPU inference and benchmarking
```

1. Train the FP32 model and save its `state_dict`.
2. Evaluate the checkpoint and save the baseline metrics.
3. Export the model to a PT2E graph with `torch.export`.
4. Insert observers using `prepare_pt2e` and `X86InductorQuantizer`.
5. Calibrate with 100 batches (400 CIFAR-10 training images).
6. Convert the calibrated graph and evaluate it on the full test set.
7. Compile the quantized graph with TorchInductor and compare compiled and uncompiled outputs.
8. Benchmark FP32 and compiled INT8 inference on the CPU.

## Current Results

| Metric | FP32 | PT2E INT8 |
| --- | ---: | ---: |
| CIFAR-10 test accuracy | 54.1% | 54.2% |
| Difference from FP32 | - | +0.1 percentage points |
| Raw layer-weight storage | 270,176 bytes | 67,544 bytes |

The raw weight comparison gives a `4.0x` storage ratio because FP32 values use four bytes and INT8 values use one. It is not a measurement of a final serialized PT2E model. The FP32 checkpoint is 0.261 MB and the network has 67,642 parameters.

The uncompiled quantized graph currently runs and returns FP32 logits with shape `[4, 10]`. The `torch.compile` path and output comparison are implemented, but a fresh local run cannot reach compiled inference because `cl.exe` is not currently available. CPU latency benchmarking is currently in progress.

## Repository Structure

- `src/model.py` - defines `SmallCNN`.
- `src/train.py` - trains the FP32 model and saves its checkpoint.
- `src/evaluate.py` - evaluates the checkpoint and writes FP32 metrics.
- `src/quantize.py` - exports, calibrates, converts, evaluates, and compiles the PT2E graph.
- `src/benchmark.py` - placeholder for the CPU benchmark.
- `results/` - saved experiment metrics.
- `checkpoints/` - local model checkpoints (ignored by Git).
- `daily-log.md` - learning and engineering progress.
- `hardware.md` - recorded test environment.

## Environment / CPU Quantization

The project uses PyTorch, torchao, PT2E, `X86InductorQuantizer`, and TorchInductor through `torch.compile`. On Windows, TorchInductor CPU compilation also needs an MSVC x64 C++ toolchain and UTF-8 mode (`PYTHONUTF8=1`).

Run the main scripts from the repository root:

```powershell
python src/train.py
python src/evaluate.py
python src/quantize.py
```

## Next Step

- Restore the MSVC x64 compiler environment and re-check compiled INT8 output.
- Benchmark batch-1 latency and throughput for FP32 and compiled INT8 inference.
- Save the benchmark results.
