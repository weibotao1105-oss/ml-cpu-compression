# Project Progress Log

This file records the main stages of the project. It is organised by technical milestones rather than by individual days because the work was completed across multiple sessions.

## 1. Project Setup and CIFAR-10 Baseline

### Work completed

- Reviewed the introductory machine-learning concepts.
- Initialized the Git repository.
- Created an isolated Python environment.
- Installed and verified the main dependencies.
- Recorded the hardware and software environment.
- Created the initial repository structure.
- Downloaded and loaded CIFAR-10 in batches with PyTorch.
- Inspected image tensors, RGB channels, labels, and class names.

This stage helped me understand the dataset, `DataLoader`, and how images are represented as tensors before they enter the model.

## 2. Building and Understanding SmallCNN

### Work completed

- Built a small CNN layer by layer with convolution, ReLU, max pooling, flattening, and fully connected layers.
- Traced tensor shapes through the complete forward pass.
- Computed cross-entropy loss and used backpropagation to inspect gradients.
- Verified an SGD update by comparing a convolution weight before and after `optimizer.step()`.
- Calculated accuracy across the 50,000-image training set with gradient tracking disabled.
- Moved the CNN into a reusable `SmallCNN` class in `model.py`.

The final model has two convolution blocks followed by a fully connected classifier. Its small size makes the later quantisation steps easier to inspect.

## 3. FP32 Training and Evaluation

### Work completed

- Trained the network for two complete epochs.
- Updated the training script to use the model class and a single epoch loop.
- Saved the trained FP32 weights as a checkpoint.
- Added separate training-set and test-set accuracy checks.
- Added an evaluation script that reloads the checkpoint and uses `model.eval()`.
- Calculated FP32 model size and parameter count.
- Added JSON output for the FP32 evaluation metrics.

### Results

- Recorded the FP32 baseline: 54.1% test accuracy, a 0.261 MB checkpoint and 67,642 parameters.

## 4. Manual Quantisation Experiments

### Work completed

- Checked the first convolution activation range over 100 calibration batches.
- Calculated symmetric per-layer scales for convolution and linear weights.
- Converted the four weight tensors from FP32 to `torch.int8`.
- Confirmed that the raw INT8 weight tensors use one quarter of the FP32 weight storage (`4.0x` compression).
- Explored FP32-to-INT8 mapping, dequantisation, and quantisation error before using the PT2E tools.

### Limitation at this stage

- The manually created INT8 tensors were not used for inference. They were a learning experiment rather than the final quantised model.

## 5. PT2E Post-Training Quantisation

### Work completed

- Replaced the manual-only experiment with an end-to-end PT2E static quantization path.
- Exported the trained `SmallCNN` with `torch.export` and configured `X86InductorQuantizer`.
- Used `prepare_pt2e` to insert observers, then calibrated them with 100 batches (400 CIFAR-10 training images).
- Converted the calibrated graph with `convert_pt2e` and ran quantized inference.
- Evaluated both the FP32 and PT2E INT8 graphs on all 10,000 CIFAR-10 test images.

### What I learned

- Weight quantization can use statistics from fixed model parameters, while activation quantization needs representative calibration data.
- `prepare_pt2e` inserts observers; calibration collects activation ranges; `convert_pt2e` rewrites the graph with quantize and dequantize operations.
- A quantized model can use INT8 operations internally while still returning FP32 logits.

### Results

- FP32 test accuracy: `54.1%`.
- PT2E INT8 test accuracy: `54.2%`.
- Difference from FP32: `+0.1` percentage points for INT8.
- The quantized graph returned FP32 output with shape `[4, 10]`.
- Raw layer weights require 270,176 bytes in FP32 and 67,544 bytes in INT8, a `4.0x` ratio. This is not a serialized INT8 model-size result.

## 6. Compiled CPU Inference

### Work completed

- Added a `torch.compile` path and a numerical comparison between compiled and uncompiled quantized outputs.
- Added compilation of the FP32 model so the benchmark compares compiled FP32 with compiled INT8.
- Added an ignored VS Code terminal profile that enables UTF-8 mode, activates the project `.venv`, and attempts to initialize the MSVC x64 environment.

### Problems / fixes

- TorchInductor previously encountered Windows GBK decoding problems, so the project terminal enables `PYTHONUTF8=1`.
- The terminal profile was written to initialize MSVC and the project `.venv` together.
- A previous fresh verification stopped when `cl.exe` was unavailable. The later benchmark JSON files show that compiled FP32 and INT8 inference completed in the environment used for those recorded runs.

## 7. Dynamic Batch Shapes

### Work completed

- Changed the original fixed batch-size export to use `torch.export.Dim`.
- Set the supported dynamic batch range to 1 through 32.
- Used the same exported graph for benchmark batch sizes `1`, `4`, `8`, `16`, and `32`.

This allows several batch sizes to use one quantised and compiled graph instead of exporting a new graph each time.

## 8. CPU Benchmarking

### Work completed

- Added `benchmark_model()`, `latency_comparison()`, `take_batches()`, and `results_collect()` helper functions.
- Used `time.perf_counter` for timing.
- Added 20 warm-up calls followed by 1,000 timed calls for each measurement.
- Repeated the FP32 and INT8 measurements five times for every batch size.
- Calculated mean and median batch latency in milliseconds.
- Calculated throughput in images per second.
- Stored results in JSON and added timestamped filenames for separate program runs.

There are currently six saved benchmark runs. Each contains FP32 and INT8 measurements for all five tested batch sizes.

## 9. Statistical Analysis of Benchmark Variability

### Work completed

- Added JSON loading across the saved benchmark history.
- Added mean, median, and sample standard deviation calculations.
- Paired corresponding FP32 and INT8 runs with `zip()`.
- Calculated the INT8 latency slowdown separately for each run before summarising the slowdown values.

### Current observation

- INT8 accuracy remains close to FP32 accuracy.
- All 30 stored batch/run comparisons currently show higher INT8 latency and lower INT8 throughput than FP32.
- Mean throughput increases with batch size for both models.
- Mean INT8 slowdown varies by batch size, from `16.50%` to `32.13%` across the six saved runs.
- The results do not prove the cause. Quantise/dequantise overhead, the small model size, and backend kernel behaviour are possible explanations to investigate.

### Next step

- Run and save the cross-run statistics for every batch size.
- Produce latency and throughput figures and a final summary table.
- Move benchmark and statistics code into `src/benchmark.py` and organise the runnable pipeline with helper functions and `main()`.
