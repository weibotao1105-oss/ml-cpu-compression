import os
from matplotlib import pyplot as plt
import torch
from model import SmallCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = SmallCNN()

saved_weights = torch.load(
    "checkpoints/fp32_model.pth"
)
# 把之前训练完成模型的参数覆盖到新建的模型
model.load_state_dict(saved_weights)

model.eval()

print("Model loaded successfully")

print(
    "conv1 weights match:",
    torch.equal(
        saved_weights["conv1.weight"],
        model.conv1.weight
    )
)

# 把图片转换成PyTorch的Tensor, 并把像素值从0-255缩放到0-1
transform = transforms.ToTensor()

# 图片和标签本身
# daataset 以图片和标签的组合形式存在为每个元素
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
# 一批一批把数据拿出来
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total

print("Test accuracy (%):", test_accuracy * 100)

# 直接拿出一张图片和一个label
image, label = test_dataset[0]

with torch.no_grad():
    output = model(image.unsqueeze(0)) # 在原来[3, 32, 32]的图片之前加一个batch维度[1, 3, 32, 32]
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

# 查看第一张图片每个类别的分数
print("\nClass scores:")

for i, class_name in enumerate(test_dataset.classes):
    print(
        f"{class_name}:"
        f"{probabilities[0][i].item() * 100:.2f}%"
    )

print("Actual:", test_dataset.classes[label])
print("Predicted:", test_dataset.classes[predicted_class])

plt.imshow(image.permute(1, 2, 0))
plt.title(
    f"Actual: {test_dataset.classes[label]} | "
    f"Predicted: {test_dataset.classes[predicted_class]}"
)
plt.show()

# 得到文件大小,单位是bytes
model_size = os.path.getsize("checkpoints/fp32_model.pth")

print(
    f"FP32 model size: {model_size / (1024 * 1024):.2f} MB"
)

num_parameters = sum(
    p.numel() # parameter tensorl里一共有多少个数字
    for p in model.parameters()
)

# FP32 -> 32 bit = 4 bytes per parameters
print("Number of parameters:", num_parameters)

# 用json保存数据
import json

fp32_results = {
    "test_accuracy": test_accuracy * 100,
    "model_size_mb": model_size / (1024 * 1024),
    "num_parameters": num_parameters
}

with open("results/fp32_metrics.json", "w") as f:
    json.dump(fp32_results, f, indent=4)

print("FP32 metrics saved.")
