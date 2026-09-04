# ML CPU Compression

This project is a small experiment to understand how post-training INT8 quantisation affects CNN inference on a CPU. I use a simple CIFAR-10 model so I can focus on the quantisation and systems side rather than model complexity.

## Goal

The project compares an FP32 model with a PT2E INT8 model across:

- CIFAR-10 classification accuracy;
- raw weight representation;
- compiled CPU batch latency;
- throughput in images per second;
- behaviour at different batch sizes.

It is mainly a learning project about efficient ML inference and how measured performance can differ from the expected theoretical improvement.

## Model and Dataset

`SmallCNN` is trained and evaluated on CIFAR-10. It contains two convolution, ReLU, and max-pooling blocks, followed by a 64-unit fully connected layer and a 10-class output layer.

## Project Flow

```text
CIFAR-10
-> SmallCNN
-> FP32 training and evaluation
-> save FP32 checkpoint
-> manual quantisation experiments
-> torch.export with a dynamic batch dimension
-> PT2E prepare and calibration
-> INT8 conversion
-> torch.compile
-> compiled FP32 and INT8 CPU inference
-> latency and throughput benchmarks
-> repeated-run statistics
```

The main stages are training the baseline, understanding the quantisation maths, building the PT2E graph, compiling both models, and then comparing repeated measurements across batch sizes.

## Quantisation Pipeline

I first used manual FP32-to-INT8 experiments to understand scaling, rounding, dequantisation error, layer-specific ranges, and the theoretical `4.0x` reduction in raw weight storage.

The end-to-end path uses torchao PT2E. `torch.export` creates the graph, `X86InductorQuantizer` supplies the x86 CPU configuration, and `prepare_pt2e` inserts observers. Calibration passes 100 batches (400 CIFAR-10 training images) through those observers without retraining. `convert_pt2e` then produces the quantised graph, which is compiled with TorchInductor through `torch.compile`.

## Benchmark Method

- Both the FP32 and INT8 graphs are compiled with `torch.compile`.
- The exported graph supports batch sizes from 1 to 32; the tested sizes are `1`, `4`, `8`, `16`, and `32`.
- FP32 and INT8 use the same input tensor for each comparison.
- Each measurement has 20 warm-up calls followed by 1,000 timed inference calls.
- Each batch size is measured five times per program run.
- Batch latency is reported in milliseconds and throughput is reported in images per second.
- Runs are saved as JSON, including timestamped files so separate executions can be compared.

Helpers for cross-run mean, median, standard deviation, and paired slowdown calculations are implemented. Producing one automatic summary for every batch size is still in progress.

## Current Results

| Accuracy metric | FP32 | PT2E INT8 |
| --- | ---: | ---: |
| CIFAR-10 test accuracy | 54.1% | 54.2% |
| Difference from FP32 | - | +0.1 percentage points |

The network has 67,642 parameters. Its four layer-weight tensors use 270,176 bytes in FP32 and 67,544 bytes in INT8; this is a raw tensor comparison, not a final serialised INT8 model-size measurement.

The table below summarises the mean of the six stored benchmark runs:

| Batch | FP32 latency (ms) | INT8 latency (ms) | FP32 images/s | INT8 images/s | Mean INT8 slowdown |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1534 | 0.2015 | 6,600 | 4,993 | 32.13% |
| 4 | 0.2506 | 0.2916 | 16,056 | 13,780 | 16.50% |
| 8 | 0.3090 | 0.3686 | 26,011 | 22,071 | 18.87% |
| 16 | 0.4211 | 0.4929 | 38,540 | 32,950 | 17.04% |
| 32 | 0.5286 | 0.6756 | 60,930 | 47,538 | 28.07% |

So far, compiled FP32 has lower latency and higher throughput in every stored comparison. The size of the INT8 slowdown changes between independent runs, so this is a preliminary observation rather than a final conclusion. Possible reasons include quantise/dequantise overhead, the small model size, and CPU kernel or operator behaviour, but these have not been proven yet.

## Repository Structure

- `src/model.py` - defines `SmallCNN`.
- `src/train.py` - trains the FP32 model and saves its checkpoint.
- `src/evaluate.py` - evaluates the checkpoint and writes FP32 metrics.
- `src/quantize.py` - contains PT2E conversion, compilation, benchmarking, and current statistics work.
- `src/benchmark.py` - placeholder for separating benchmark code later.
- `results/` - FP32 metrics and repeated benchmark JSON files.
- `checkpoints/` - local model checkpoints, ignored by Git.
- `progress-log.md` - milestone-based learning and engineering log.
- `hardware.md` - recorded test environment.

## Environment

The project uses CPU-only PyTorch, torchao, PT2E, `X86InductorQuantizer`, and TorchInductor. On Windows, compiled CPU execution requires UTF-8 mode and an MSVC x64 C++ toolchain with `cl.exe` available in the terminal environment.

## Current Status / Next Step

The project is now at the benchmark-analysis stage. The next step is to finish the cross-run summary for all batch sizes, save a final results table, produce latency and throughput figures, and then move the benchmark-specific code into `src/benchmark.py` without changing the experiment itself.
