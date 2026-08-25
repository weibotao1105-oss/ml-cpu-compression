import torch
from model import SmallCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 让数据通过conv1然后查看activation 过后的shape

model = SmallCNN()

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

# 使用更多的图片做calibration, 因为量化activation与weight不同， weight随机生成提取一次min 和max 就可以，而activation对不同的图片input出来的范围不一样
# 先把最小值设成“正无穷”，这样第一批数据的任何真实最小值都会比它小
activation_min = float("inf")
# 同理对最大值
activation_max = float("-inf")

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(calibration_loader):

        activation = model.conv1(images)

        batch_min = activation.min().item()
        batch_max = activation.max().item()

        activation_min = min(
            activation_min,
            batch_min
        )

        activation_max = max(
            activation_max,
            batch_max
        )

        if batch_index == 99:
            break

print("Calibration activation min:", activation_min)
print("Calibration activation max:", activation_max)

# 计算scale
activation_max_abs = max(
    abs(activation_min),
    abs(activation_max)
)

activation_scale = activation_max_abs / 127

print("Activation maximum absolute value:", activation_max_abs)
print("Activation scale:", activation_scale)

#
model.load_state_dict(
    torch.load("checkpoints/fp32_model.pth")
)

model.eval()

print("conv1 weight dtype:", model.conv1.weight.dtype)
print("fc1 weight dtype:", model.fc1.weight.dtype)

weights = model.conv1.weight

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
def calculate_scale(weights):
    min_weight = weights.min().item()
    max_weight = weights.max().item()

    max_abs = max(
        abs(min_weight),
        abs(max_weight)
    )

    scale = max_abs / 127

    return scale

conv1_scale = calculate_scale(model.conv1.weight)
conv2_scale = calculate_scale(model.conv2.weight)
fc1_scale = calculate_scale(model.fc1.weight)
fc2_scale = calculate_scale(model.fc2.weight)

print("conv1 scale:", conv1_scale)
print("conv2 scale:", conv2_scale)
print("fc1 scale:", fc1_scale)
print("fc2 scale:", fc2_scale)

# 量化函数
def quantize_weights(weights, scale):

    quantized = torch.round(
        weights / scale
    )
    # 将数据clamp在-127到127之间
    quantized = quantized.clamp(
        -127,
        127
    )

    quantized = quantized.to(torch.int8)

    return quantized

conv1_quantized = quantize_weights(
    model.conv1.weight,
    conv1_scale
)

conv2_quantized = quantize_weights(
    model.conv2.weight,
    conv2_scale
)

fc1_quantized = quantize_weights(
    model.fc1.weight,
    fc1_scale
)

fc2_quantized = quantize_weights(
    model.fc2.weight,
    fc2_scale
)

fp32_weight_bytes = sum(
    layer.weight.numel() * layer.weight.element_size()
    for layer in [
        model.conv1,
        model.conv2,
        model.fc1,
        model.fc2
    ]
)

int8_weight_bytes = sum(
    weights.numel() * weights.element_size()
    for weights in [
        conv1_quantized,
        conv2_quantized,
        fc1_quantized,
        fc2_quantized
    ]
)

print("FP32 weight size:", fp32_weight_bytes, "bytes")
print("INT8 weight size:", int8_weight_bytes, "bytes")
print("Compression ratio:", fp32_weight_bytes / int8_weight_bytes)
