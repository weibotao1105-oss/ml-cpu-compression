# import torch
# from model import SmallCNN
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # 让数据通过conv1然后查看activation 过后的shape

# model = SmallCNN()

# transform = transforms.ToTensor()

# calibration_dataset = datasets.CIFAR10(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform
# )

# calibration_loader = DataLoader(
#     calibration_dataset,
#     batch_size=4,
#     shuffle=False
# )

# # 使用更多的图片做calibration, 因为量化activation与weight不同， weight随机生成提取一次min 和max 就可以，而activation对不同的图片input出来的范围不一样
# # 先把最小值设成“正无穷”，这样第一批数据的任何真实最小值都会比它小
# activation_min = float("inf")
# # 同理对最大值
# activation_max = float("-inf")

# with torch.no_grad():

#     for batch_index, (images, labels) in enumerate(calibration_loader):

#         activation = model.conv1(images)

#         batch_min = activation.min().item()
#         batch_max = activation.max().item()

#         activation_min = min(
#             activation_min,
#             batch_min
#         )

#         activation_max = max(
#             activation_max,
#             batch_max
#         )

#         if batch_index == 99:
#             break

# print("Calibration activation min:", activation_min)
# print("Calibration activation max:", activation_max)

# # 计算scale
# activation_max_abs = max(
#     abs(activation_min),
#     abs(activation_max)
# )

# activation_scale = activation_max_abs / 127

# print("Activation maximum absolute value:", activation_max_abs)
# print("Activation scale:", activation_scale)

# #
# model.load_state_dict(
#     torch.load("checkpoints/fp32_model.pth")
# )

# model.eval()

# print("conv1 weight dtype:", model.conv1.weight.dtype)
# print("fc1 weight dtype:", model.fc1.weight.dtype)

# example_inputs = (
#     torch.randn(4, 3, 32, 32),
# )

# exported_model = torch.export.export(
#     model,
#     example_inputs
# ).module()

# print("Model exported successfully.")
# print(type(exported_model))

# weights = model.conv1.weight

# # item(), 把单个数值的tensor取成普通python 数字
# # 找到weight 里最大和最小的然后映射到INT8的-127-127范围内
# print("conv1 minimum weight:", weights.min().item())
# print("conv1 maximum weight:", weights.max().item())

# # 计算得到scale, INT8和FP32的对应关系
# max_abs = max(
#     abs(weights.min().item()),
#     abs(weights.max().item())
# )

# scale = max_abs / 127

# print("Maximum absolute weight:", max_abs)
# print("Scale:", scale)

# # 手动把它量化成INT8，由于INT8只能存整数，所以还原的时候会与之前的数字结果不一样，所以要计算误差
# first_weight = weights[0, 0, 0, 0].item()

# quantized_weight = round(first_weight / scale)

# dequantized_weight = quantized_weight * scale

# print("Original FP32 weight:", first_weight)
# print("Quantized INT8 value:", quantized_weight)
# print("Dequantized value:", dequantized_weight)

# quantization_error = abs(
#     first_weight - dequantized_weight
# )

# print("Quantization error:", quantization_error)

# relative_error = quantization_error / abs(first_weight)

# print("Relative error (%):", relative_error * 100)

# # 把整个conv1.weight tensor一次性量化成INT8
# quantized_weights = torch.round(
#     weights / scale  # 对weight 里的每一个数字都除以同一个scale
# ).to(torch.int8)

# print("Quantized weights dtype:", quantized_weights.dtype)
# print("Quantized weights shape:", quantized_weights.shape)

# print("First few quantized weights:")
# print(quantized_weights[0])

# # 反量化找误差
# dequantized_weights = quantized_weights.float() * scale  # float() 把原本的INT8数据类型转换为浮点表示以便后面乘以scale恢复近似的FP32 weight

# absolute_errors = torch.abs(
#     weights - dequantized_weights
# )

# mean_error = absolute_errors.mean().item()
# max_error = absolute_errors.max().item()

# print("Mean quantization error:", mean_error)
# # 最大误差大约是scale/2, 因为我们是“四舍五入到最近的整数”，所以最坏情况下，一个数最多距离最近整数
# print("Maximum quantization error:", max_error)

# theoretical_max_error = scale / 2

# print("Scale / 2:", theoretical_max_error)
# print("Actual max error:", max_error)

# # 转换fc1 层的weight
# fc1_weights = model.fc1.weight

# fc1_min = fc1_weights.min().item()
# fc1_max = fc1_weights.max().item()

# fc1_max_abs = max(
#     abs(fc1_min),
#     abs(fc1_max)
# )

# fc1_scale = fc1_max_abs / 127

# print("\nfc1 minimum weight:", fc1_min)
# print("fc1 maximum weight:", fc1_max)
# print("fc1 maximum absolute weight:", fc1_max_abs)
# print("fc1 scale:", fc1_scale)
# print("conv1 scale:", scale)

# 将计算scale整合为一个函数
# def calculate_scale(weights):
#     min_weight = weights.min().item()
#     max_weight = weights.max().item()

#     max_abs = max(
#         abs(min_weight),
#         abs(max_weight)
#     )

#     scale = max_abs / 127

#     return scale

# conv1_scale = calculate_scale(model.conv1.weight)
# conv2_scale = calculate_scale(model.conv2.weight)
# fc1_scale = calculate_scale(model.fc1.weight)
# fc2_scale = calculate_scale(model.fc2.weight)

# print("conv1 scale:", conv1_scale)
# print("conv2 scale:", conv2_scale)
# print("fc1 scale:", fc1_scale)
# print("fc2 scale:", fc2_scale)

# # 量化函数
# def quantize_weights(weights, scale):

#     quantized = torch.round(
#         weights / scale
#     )
#     # 将数据clamp在-127到127之间
#     quantized = quantized.clamp(
#         -127,
#         127
#     )

#     quantized = quantized.to(torch.int8)

#     return quantized

# conv1_quantized = quantize_weights(
#     model.conv1.weight,
#     conv1_scale
# )

# conv2_quantized = quantize_weights(
#     model.conv2.weight,
#     conv2_scale
# )

# fc1_quantized = quantize_weights(
#     model.fc1.weight,
#     fc1_scale
# )

# fc2_quantized = quantize_weights(
#     model.fc2.weight,
#     fc2_scale
# )

# fp32_weight_bytes = sum(
#     layer.weight.numel() * layer.weight.element_size()
#     for layer in [
#         model.conv1,
#         model.conv2,
#         model.fc1,
#         model.fc2
#     ]
# )

# int8_weight_bytes = sum(
#     weights.numel() * weights.element_size()
#     for weights in [
#         conv1_quantized,
#         conv2_quantized,
#         fc1_quantized,
#         fc2_quantized
#     ]
# )

# print("FP32 weight size:", fp32_weight_bytes, "bytes")
# print("INT8 weight size:", int8_weight_bytes, "bytes")
# print("Compression ratio:", fp32_weight_bytes / int8_weight_bytes)
from datetime import datetime
import glob
import statistics

import torch
from model import SmallCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = SmallCNN()

model.load_state_dict(
    torch.load("checkpoints/fp32_model.pth")
)

model.eval()

example_inputs = (
    torch.randn(4, 3, 32, 32),
)

# 应用dynamic shape

batch_dim = torch.export.Dim(
    "batch",
    min=1,
    max=32
)

exported_model = torch.export.export(
    model,
    example_inputs,
    dynamic_shapes={
        "x": {0: batch_dim}  #对于输入 x，把它的第 0 维设置成一个允许 1～32 变化的动态维度
    }
).module()

print("Model exported successfully.")
print(type(exported_model))

from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e
)

import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq

from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer
)

# Quantization math (FP32 -> INT8) is mostly hardware-independent,
# but efficient INT8 inference depends on the CPU backend and its supported instructions/kernels.

#告诉 PyTorch，我们最终想在x86CPU上运行量化模型
quantizer = X86InductorQuantizer()

# 获取x86后端的默认量化配置
quantizer.set_global(
    xiq.get_default_x86_inductor_quantization_config()
)

# 在模型graph中适当的位置插入observers
# observer 可看作一个测量仪，在图片跑过模型的时候，这些observer会记录：
# 这里的 activation 大概最小是多少？
# 最大是多少？
# 分布范围是什么？
prepared_model = prepare_pt2e(
    exported_model,
    quantizer
)

transform = transforms.ToTensor()

calibration_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=4,
    shuffle=False
)

with torch.no_grad():
    # 这里CIFAR-10 每个 sample 都会给image + label
    for batch_index, (images, _) in enumerate(calibration_loader):

        prepared_model(images)

        if batch_index == 99:
            break

print("Calibration completed.")

# 会读取 calibration 阶段 observers 收集到的统计信息，然后把原来的 prepared graph 改写成带有 quantize/dequantize 操作的 quantized graph
quantized_model = convert_pt2e(prepared_model)

print("Model converted to quantized graph.")

# 从loader里拿一个batch, _ 仍然表示 label 这里暂时不用。
test_images, _ = next(iter(calibration_loader))

with torch.no_grad():
    quantized_output = quantized_model(test_images)

print("Quantized output shape:", quantized_output.shape)

# 模型内部可以使用 INT8 quantization，但模型最终输出 logits 仍然可以是 FP32
print("Quantized output dtype:", quantized_output.dtype)

# Calibration only collects activation statistics; convert_pt2e() uses them to rewrite the model into a quantized graph.

# 测量完整 INT8 quantized model 在 10,000 张 CIFAR-10 test images 上的 accuracy
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

# 提取最初FP32模型
fp32_correct = 0
fp32_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        output = model(images)

        predicted = torch.argmax(output, dim=1)

        fp32_correct += (predicted == labels).sum().item()
        fp32_total += labels.size(0)

fp32_accuracy = fp32_correct / fp32_total * 100

# INT8模型
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:
        # 这里不再使用原始 FP32 model，而是你刚刚 convert_pt2e() 得到的量化模型
        output = quantized_model(images)

        predicted = torch.argmax(output, dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

quantized_accuracy = correct / total


int8_accuracy = quantized_accuracy * 100

accuracy_drop = fp32_accuracy - int8_accuracy

print("FP32 accuracy (%):", fp32_accuracy)
print("INT8 accuracy (%):", int8_accuracy)
print("Accuracy drop (percentage points):", accuracy_drop)

# 开始真正针对CPU编译
compiled_int8_model = torch.compile(quantized_model)

with torch.no_grad():
    compiled_output = compiled_int8_model(test_images)

print("Compiled output shape:", compiled_output.shape)

with torch.no_grad():
    uncompiled_output = quantized_model(test_images)

difference = torch.abs(
    compiled_output - uncompiled_output
)

print("Maximum output difference:", difference.max().item())
print(
    "Outputs close:",
    torch.allclose(
        compiled_output,
        uncompiled_output,
        atol=1e-5
    )
)
# FP32 compiled model
compiled_fp32_model = torch.compile(model)

import time

def benchmark_model(model, input_tensor):
    # Warm up
    with torch.no_grad():
        for _ in range(20):
            model(input_tensor)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(1000):
            model(input_tensor)

    end = time.perf_counter()

    average_time = (end - start) / 1000

    return average_time

def latency_comparison(fp32model, int8model, input, batch_size):
    fp32_latency_list = []
    int8_latency_list = []
    for run in range(5):
        fp32_time = benchmark_model(
            fp32model,
            input
        )
        fp32_latency_list.append(fp32_time)

        int8_time = benchmark_model(
            int8model,
            input
        )
        int8_latency_list.append(int8_time)

        print(
            f"Run {run+1}:"
            f"FP32 = {fp32_time * 1000:.4f} ms,"
            f"INT8 = {int8_time * 1000:.4f} ms"
        )

    fp32_latency_mean = (sum(fp32_latency_list) / len(fp32_latency_list)) * 1000
    int8_latency_mean = (sum(int8_latency_list) / len(int8_latency_list)) * 1000

    fp32_median = statistics.median(fp32_latency_list) * 1000
    int8_median = statistics.median(int8_latency_list) * 1000


    fp32_throughput = batch_size/(fp32_latency_mean/1000)
    int8_throughput = batch_size/(int8_latency_mean/1000)

    print(f"FP32 batch size {batch_size} mean latency: {fp32_latency_mean:.4f} ms")
    print(f"INT8 batch size {batch_size} mean latency: {int8_latency_mean:.4f} ms")

    print(f"FP32 latency median: {fp32_median:.2f} ms")
    print(f"INT8 latency median: {int8_median:.2f} ms")

    print(f"INT8 slower by:{(int8_latency_mean - fp32_latency_mean) / fp32_latency_mean * 100:.4f} %")

    print(f"FP32 throughput: {fp32_throughput:.2f} per second")
    print(f"INT8 throughput: {int8_throughput:.2f} per second")

    return fp32_latency_mean, fp32_median, int8_latency_mean, int8_median, fp32_throughput, int8_throughput

def take_batches(n):
    batch_list = []
    loader_iter = iter(test_loader)

    for _ in range(n):
        batch = next(loader_iter)
        images = batch[0]
        batch_list.append(images)

    combined_batch = torch.cat(batch_list, dim=0)
    return combined_batch

def results_collect(fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput, batch_size, result_list):
    data_dic = {
        "batch_size": batch_size,
        "fp32_latency_mean_ms" : fp32_latency_mean,
        "fp32_latency_median_ms" : fp32_latency_median,
        "fp32_throughput": fp32_throughput,
        "int8_latency_mean_ms" : int8_latency_mean,
        "int8_latency_median_ms" : int8_latency_median,
        "int8_throughput": int8_throughput
    }
    result_list.append(data_dic)

results_list = []
# batch size = 1
batch1_input = test_images[:1]
print("Batch size = 1:")
fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput = latency_comparison(compiled_fp32_model, compiled_int8_model, batch1_input, 1)
results_collect(fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput, 1, results_list)

# batch size = 4
size4_batch = take_batches(1)
print("Batch size = 4:")
fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput = latency_comparison(compiled_fp32_model, compiled_int8_model, size4_batch, 4)
results_collect(fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput, 4, results_list)

# batch size = 8
size8_batch = take_batches(2)
print("Batch size = 8:")
fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput = latency_comparison(compiled_fp32_model, compiled_int8_model, size8_batch, 8)
results_collect(fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput, 8, results_list)

# batch size = 16
print("Batch size = 16")
size16_batch = take_batches(4)
fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput = latency_comparison(compiled_fp32_model, compiled_int8_model, size16_batch, 16)
results_collect(fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput, 16, results_list)

# batch size = 32
print("Batch size = 32")
size32_batch = take_batches(8)
fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput = latency_comparison(compiled_fp32_model, compiled_int8_model, size32_batch, 32)
results_collect(fp32_latency_mean, fp32_latency_median, int8_latency_mean, int8_latency_median, fp32_throughput, int8_throughput, 32, results_list)
print(results_list)

import json

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

filename = f"results/benchmark_{timestamp}.json"

with open(filename, "w") as file:
    json.dump(results_list, file, indent=4)

# 返回值是一个list
files = glob.glob("results/benchmark_*.json")
files = sorted(files)

with open(files[0], "r") as file:
    data = json.load(file)

def json_data_load(files, batch_size, stats_type):
    fp32_runs = []
    int8_runs = []
    fp32_key = f"fp32_{stats_type}"
    int8_key = f"int8_{stats_type}"

    for filename in files:
        with open(filename, "r") as file:
            data = json.load(file)
        for results in data:
            if results["batch_size"] == batch_size:
                fp32_runs.append(results[fp32_key])
                int8_runs.append(results[int8_key])
    return fp32_runs, int8_runs

def generate_stats(files, batch_size, stats_type):
    fp32_runs, int8_runs = json_data_load(files, batch_size, stats_type)

    fp32_mean = statistics.mean(fp32_runs)
    fp32_median = statistics.median(fp32_runs)
    fp32_std = statistics.stdev(fp32_runs)

    int8_mean = statistics.mean(int8_runs)
    int8_median = statistics.median(int8_runs)
    int8_std = statistics.stdev(int8_runs)

    print(f"FP32 {stats_type}:\n{fp32_mean:.2f}\n{fp32_median:.2f}\n{fp32_std:.2f}")

    print(f"INT8 {stats_type}:\n{int8_mean:.2f}\n{int8_median:.2f}\n{int8_std:.2f}")

def slowdown_percentage(files, batch_size, stats_type):
    fp32_runs, int8_runs = json_data_load(files, batch_size, stats_type)
    slowdown_percentages = []

    for fp32_time, int8_time in zip(fp32_runs, int8_runs):
        slow_p = ((int8_time - fp32_time) / fp32_time) * 100
        slowdown_percentages.append(slow_p)

    slow_mean = statistics.mean(slowdown_percentages)
    slow_median = statistics.median(slowdown_percentages)
    slow_std = statistics.stdev(slowdown_percentages)

    print(f"INT8 slowdown stats:\n mean: {slow_mean:.2f}%\n median: {slow_median:.2f}%\n std: {slow_std:.2f}%")

    return slow_mean, slow_median, slow_std, slowdown_percentages
