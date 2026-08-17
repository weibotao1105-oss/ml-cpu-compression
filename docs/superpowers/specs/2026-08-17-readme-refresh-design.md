# README Refresh Design

## Goal

Update the repository README so it accurately describes the project, the current educational CNN milestone, how to run the code, and the remaining FP32-to-INT8 experiment roadmap.

## Audience and language

The README will remain in English and target GitHub readers who want to understand or reproduce the project. Explanations will be concise and practical rather than duplicating the step-by-step teaching comments in `src/train.py`.

## Content structure

The refreshed README will contain:

1. A corrected project title and research question.
2. A short project overview explaining the CPU quantization objective.
3. A current-progress section that records only implemented work:
   - CIFAR-10 loading and batching.
   - A manually assembled small CNN forward path.
   - Cross-entropy loss, backpropagation, and SGD weight updates.
   - Two training epochs and full-training-set accuracy calculation.
4. The current model architecture: `3 -> 8 -> 16` convolution channels, two `2 x 2` max-pooling stages, flattening to 1,024 features, and `1024 -> 64 -> 10` fully connected layers.
5. Environment setup and a command for running `src/train.py` on Windows.
6. A repository-layout section that describes files that actually exist.
7. A roadmap that clearly labels test-set evaluation, model refactoring, FP32 baseline measurements, INT8 quantization, and CPU benchmarking as future work.

## Accuracy and scope rules

- Do not invent training accuracy, test accuracy, latency, throughput, or model-size results.
- Do not present the current training script as a finished production training pipeline; it is intentionally educational and verbose.
- Keep completed milestones separate from planned work.
- Fix spelling, Markdown list formatting, and outdated repository-structure claims in the existing README.
- Do not change source code, dependency versions, or experiment behavior as part of this documentation update.

## Verification

Before committing the README, verify that:

- every claimed completed feature is present in the current source;
- every unfinished experiment is labeled as planned;
- referenced paths and the run command match the repository;
- Markdown has no obvious structural or whitespace errors;
- the final Git diff includes only the intended documentation plus the user's existing source changes.

## Delivery

After approval of this design, create an implementation plan, update `README.md`, run proportionate checks, commit all current changes on `main`, and push `main` to `origin` without force-pushing.
