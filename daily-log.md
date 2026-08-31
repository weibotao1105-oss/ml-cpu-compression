# Daily Research Log

## Day 1

### Work completed

- Reviewed the introductory machine-learning concepts.
- Initialized the Git repository.
- Created an isolated Python environment.
- Installed and verified the main dependencies.
- Recorded the hardware and software environment.
- Created the initial repository structure.

## Day 2

### Work completed

- Downloaded and loaded CIFAR-10 in batches with PyTorch.
- Inspected image tensors, RGB channels, labels, and class names.
- Built a small CNN layer by layer with convolution, ReLU, max pooling, flattening, and fully connected layers.
- Traced tensor shapes through the complete forward pass.
- Computed cross-entropy loss and used backpropagation to inspect gradients.
- Verified an SGD update by comparing a convolution weight before and after `optimizer.step()`.
- Trained the network for two complete epochs.
- Calculated accuracy across the 50,000-image training set with gradient tracking disabled.

### Next steps

- Refactor the learning script into a reusable `nn.Module`.
- Evaluate the model on the held-out CIFAR-10 test set.
- Establish the FP32 baseline before applying INT8 quantization.

## Day 3

### Work completed

- Moved the CNN into a reusable `SmallCNN` class in `model.py`.
- Updated the training script to use the model class and a single epoch loop.
- Saved the trained FP32 weights as a checkpoint.
- Added separate training-set and test-set accuracy checks.
- Added an evaluation script that reloads the checkpoint and uses `model.eval()`.
- Calculated FP32 model size and parameter count.
- Added JSON output for the FP32 evaluation metrics.

### Next steps

- Add INT8 quantization.
- Benchmark FP32 and INT8 inference on the CPU.
- Compare accuracy, model size, latency and throughput.

## Day 4

### Work completed

- Recorded the FP32 baseline: 54.1% test accuracy, a 0.261 MB checkpoint and 67,642 parameters.
- Checked the first convolution activation range over 100 calibration batches.
- Calculated symmetric per-layer scales for convolution and linear weights.
- Converted the four weight tensors from FP32 to `torch.int8`.
- Confirmed that the raw INT8 weight tensors use one quarter of the FP32 weight storage (`4.0x` compression).

### Current limitation

- The INT8 tensors are not used for inference yet, so INT8 accuracy and latency are not available.

### Next steps

- Build an end-to-end quantized inference path.
- Measure INT8 accuracy and CPU performance.
- Compare the final FP32 and INT8 results.

## Day 5 — PTQ and compiled INT8 work

### Work completed

- Replaced the manual-only experiment with an end-to-end PT2E static quantization path.
- Exported the trained `SmallCNN` with `torch.export` and configured `X86InductorQuantizer`.
- Used `prepare_pt2e` to insert observers, then calibrated them with 100 batches (400 CIFAR-10 training images).
- Converted the calibrated graph with `convert_pt2e` and ran quantized inference.
- Evaluated both the FP32 and PT2E INT8 graphs on all 10,000 CIFAR-10 test images.
- Added a `torch.compile` path and a numerical comparison between compiled and uncompiled quantized outputs.
- Added an ignored VS Code terminal profile that enables UTF-8 mode, activates the project `.venv`, and attempts to initialize the MSVC x64 environment.

### What I learned

- Weight quantization can use statistics from fixed model parameters, while activation quantization needs representative calibration data.
- `prepare_pt2e` inserts observers; calibration collects activation ranges; `convert_pt2e` rewrites the graph with quantize and dequantize operations.
- A quantized model can use INT8 operations internally while still returning FP32 logits.
- TorchInductor needs a working native C++ compiler environment to build CPU kernels on Windows.

### Results

- FP32 test accuracy: `54.1%`.
- PT2E INT8 test accuracy: `54.2%`.
- Difference from FP32: `+0.1` percentage points for INT8.
- The quantized graph returned FP32 output with shape `[4, 10]`.
- Raw layer weights require 270,176 bytes in FP32 and 67,544 bytes in INT8, a `4.0x` ratio. This is not a serialized INT8 model-size result.

### Problems / fixes

- TorchInductor previously encountered Windows GBK decoding problems, so the project terminal enables `PYTHONUTF8=1`.
- The terminal profile was written to initialize MSVC and the project `.venv` together.
- During fresh verification, `cl.exe` was not available and `torch.compile` stopped with `InvalidCxxCompiler`. The PT2E conversion and uncompiled INT8 evaluation still completed successfully, but compiled output could not be revalidated in the current environment.

### Next step

- Restore the MSVC x64 compiler environment, rerun the compiled/uncompiled output check, and then benchmark FP32 against compiled INT8 inference.
