# README Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit the current educational CNN work, replace the outdated README with an accurate English project guide, update the daily research log, and push the resulting `main` branch to `origin`.

**Architecture:** Keep `src/train.py` as the intentionally verbose learning script and make `README.md` the concise repository entry point. Separate verified current milestones from planned FP32 and INT8 experiments, and avoid changing model behavior during this documentation task.

**Tech Stack:** Markdown, Python 3.14, PyTorch 2.13, Torchvision 0.28, Git

## Global Constraints

- The README remains in English.
- Do not invent training accuracy, test accuracy, latency, throughput, or model-size results.
- Do not present `src/train.py` as a finished production training pipeline.
- Do not change source behavior, dependency versions, or experiment behavior.
- Push `main` to `origin` without force-pushing.

---

### Task 1: Verify the published CNN learning milestone

**Files:**
- Verify only: `src/train.py`

**Interfaces:**
- Consumes: the user's educational CNN implementation in commit `e53cbc8`.
- Produces: confirmation that the published training script is present locally and syntactically valid.

- [ ] **Step 1: Compile the script without executing training**

Run: `python -m py_compile src/train.py`

Expected: exit code 0 and no syntax-error output.

- [ ] **Step 2: Verify the published commit and branch state**

Run: `git show --stat --oneline e53cbc8; git rev-parse HEAD; git rev-parse origin/main`

Expected: commit `e53cbc8` contains the educational CNN changes, and local `main` initially matches `origin/main`.

### Task 2: Replace the outdated README

**Files:**
- Modify: `README.md`
- Modify: `daily-log.md`
- Create: `docs/superpowers/plans/2026-08-17-readme-refresh.md`

**Interfaces:**
- Consumes: the implemented behavior in `src/train.py`, environment data in `hardware.md`, and project intent in the design specification.
- Produces: an English GitHub landing page with accurate setup, architecture, progress, layout, and roadmap sections, plus an English Day 2 research-log entry.

- [ ] **Step 1: Replace `README.md` with the approved content structure**

Use these exact sections and facts:

```markdown
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
```

- [ ] **Step 2: Verify README claims against the repository**

Before verification, append this entry to `daily-log.md`:

```markdown

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
```

Then run the repository checks.

Run:

```powershell
rg -n "Conv2d|MaxPool2d|Linear|CrossEntropyLoss|SGD|epoch 2|Training accuracy" src/train.py
Test-Path README.md
Test-Path daily-log.md
Test-Path hardware.md
Test-Path requirements.txt
git diff --check
```

Expected: each implementation term is found, every referenced file exists, and the whitespace check reports no errors.

- [ ] **Step 3: Inspect the final documentation diff**

Run: `git diff -- README.md daily-log.md docs/superpowers/specs/2026-08-17-readme-refresh-design.md docs/superpowers/plans/2026-08-17-readme-refresh.md; git status --short --branch`

Expected: only the approved README, daily log, and Superpowers documentation remain uncommitted.

- [ ] **Step 4: Commit the README and implementation plan**

```powershell
git add -- README.md daily-log.md docs/superpowers/specs/2026-08-17-readme-refresh-design.md docs/superpowers/plans/2026-08-17-readme-refresh.md
git commit -m "docs: update project progress and roadmap"
```

### Task 3: Verify and publish `main`

**Files:**
- Verify only: all tracked repository files

**Interfaces:**
- Consumes: the committed training milestone, README, design specification, and implementation plan.
- Produces: an up-to-date `origin/main` with no uncommitted changes.

- [ ] **Step 1: Run final local verification**

Run:

```powershell
python -m py_compile src/train.py
git diff --check HEAD
git status --short --branch
git log -5 --oneline --decorate
```

Expected: Python compilation succeeds, the whitespace check is clean, the worktree has no uncommitted changes, and the expected commits are on local `main`.

- [ ] **Step 2: Push without rewriting remote history**

Run: `git push origin main`

Expected: Git reports that `main` was updated successfully.

- [ ] **Step 3: Verify the remote ref**

Run:

```powershell
git fetch origin main
git rev-parse main
git rev-parse origin/main
git status --short --branch
```

Expected: local `main` and `origin/main` resolve to the same commit, and the worktree is clean.
