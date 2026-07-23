# CPU Inference after Neural Network Quantization

## Research question

Does converting a nerual network from FP32 to INT8 improve inference
latency, throughput and model size on a standard CPU, and what accuracy
loss does it introduce?

## Planned experiment

-Dataset: CIFAR-10
-Model: Small CNN
-Baseline: FP32 PyTorch model
-Compressed model: INT8 model
-Environment: CPU inference
-Metrics:
 -Test accuracy
 -Model size
 -Batch-1 latency
 -Batch-32 throughput

## Repository structure
- 'src/':model. training, evaluation and benchmark code
- 'notebooks/':exploratory work
- 'configs/': experiment configurations
- 'results/': figures and result tables
- 'reports/': technical report

## Current Status

Day 1: Repository and Python environment initialized.