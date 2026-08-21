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
