import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Scale -> /255
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)

print("Classes:", train_dataset.classes)

label_names = [
    train_dataset.classes[label.item()]
    for label in labels
]

print("Label names:", label_names)

print("One image shape:", images[0].shape)
print("One image tensor:", images[0])

# images.shape = [4, 3, 32, 32]
#                 ↑  ↑   ↑   ↑
#                图片通道 行  列
print("Pixel at row 0, column 0:")
print("R:", images[0, 0, 0, 0].item())
print("G:", images[0, 1, 0, 0].item())
print("B:", images[0, 2, 0, 0].item())

# Show the images
import matplotlib.pyplot as plt

image = images[0].permute(1, 2, 0) # Transform to usual matplotlib [height, width, channels] 改变排列方式

plt.imshow(image)
plt.title(label_names[0])
plt.show()

# Let these 4 images go through the first layer of CNN
import torch.nn as nn

conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=8, # CNN 第一层里有8个不同的filter，每一个filter代表一种观察方式，最终输出为8个新的feature maps
    kernel_size=3,  # 每次只看图片中的一个3*3区域
    padding=1
)

conv_output = conv1(images)

# 添加ReLU激活函数
relu = nn.ReLU()

activated_output = relu(conv_output)

print("Before ReLU min:", conv_output.min().item())
print("After ReLU min:", activated_output.min().item())

print("Before ReLU example:", conv_output[0, 0, 0, 0].item())
print("After ReLU example:", activated_output[0, 0, 0, 0].item())

# 添加Max Pooling，压缩feature maps
pool = nn.MaxPool2d(kernel_size=2, stride=2)

pooled_output = pool(activated_output)

print("Before pooling:", activated_output.shape)
print("After pooling:", pooled_output.shape)

# 加第二个convolution layer

conv2 = nn.Conv2d(
    in_channels=8,
    out_channels=16,
    kernel_size=3,
    padding=1
)

conv2_output = conv2(pooled_output)

print("Before conv2:", pooled_output.shape)
print("After conv2:", conv2_output.shape)

# 给第二层添加ReLU
relu2 = nn.ReLU()
pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

activated2 = relu2(conv2_output)
pooled2 = pool2(activated2)

print("Before pool2:", activated2.shape)
print("After pool2:", pooled2.shape)

# Flattened
flattened = torch.flatten(pooled2, start_dim=1)

print("Before flatten:", pooled2.shape)
print("After flatten:", flattened.shape)

# Linear
# 64 neurons, input vector has 1024 features, so 1024 weights
fc1 = nn.Linear(
    in_features=1024,
    out_features=64
)

fc1_output = fc1(flattened)

print("Before fc1:", flattened.shape)
print("After fc1:", fc1_output.shape)

# ReLU
fc1_activated = torch.relu(fc1_output)

# Output layer(Into classes)
fc2 = nn.Linear(
    in_features=64,
    out_features=10
)

output = fc2(fc1_activated)

print("Before fc2:", fc1_activated.shape)
print("After fc2:", output.shape)
print("First image output:", output[0])

# Find the class with the max number
predicted_class = torch.argmax(output, dim=1)

print("Predicted class indices:", predicted_class)

# Compared with the real labels
for i in range(len(predicted_class)):
    print(
        "Predicted:",
        train_dataset.classes[predicted_class[i].item()],
        "| Actual:",
        train_dataset.classes[labels[i].item()]
    )
    
# Compute the Loss
criterion = nn.CrossEntropyLoss()

loss = criterion(output, labels)

print("Loss:", loss.item())

# 初始化Optimizer 传入所有相关的weights
optimizer = torch.optim.SGD(
    list(conv1.parameters()) +
    list(conv2.parameters()) +
    list(fc1.parameters()) +
    list(fc2.parameters()),
    lr=0.01  # 步长
)

# Backpropagation
loss.backward()

print("Gradient of one conv1 weight:")
print(conv1.weight.grad[0, 0, 0, 0])

# 修改weights（Pytorch默认会累加gradient，而不是自动覆盖旧gradient)
weight_before = conv1.weight[0, 0, 0, 0].item()

optimizer.step()

weight_after = conv1.weight[0, 0, 0, 0].item()

print("Weight before update:", weight_before)
print("Weight after update:", weight_after)

# 通过loop来按4张图片一个batch来更新一次weights,一共更新12500次
print("\nStarting training loop...\n")
running_loss = 0.0
for batch_index, (images, labels) in enumerate(train_loader):

    optimizer.zero_grad() # 每次都设为0

    x = conv1(images)
    x = torch.relu(x)
    x = pool(x)

    x = conv2(x)
    x = torch.relu(x)
    x = pool2(x)

    x = torch.flatten(x, start_dim=1)

    x = fc1(x)
    x = torch.relu(x)

    output = fc2(x)

    loss = criterion(output, labels)

    loss.backward()

    optimizer.step()
    running_loss += loss.item()

average_loss = running_loss / len(train_loader)

print("Average loss for 1 epoch:", average_loss)

# Second epoch
running_loss = 0.0

for batch_index, (images, labels) in enumerate(train_loader):

    optimizer.zero_grad()

    x = conv1(images)
    x = torch.relu(x)
    x = pool(x)

    x = conv2(x)
    x = torch.relu(x)
    x = pool2(x)

    x = torch.flatten(x, start_dim=1)

    x = fc1(x)
    x = torch.relu(x)

    output = fc2(x)

    loss = criterion(output, labels)

    loss.backward()
    optimizer.step()

    running_loss += loss.item()

average_loss = running_loss / len(train_loader)

print("Average loss for epoch 2:", average_loss)

# Test Accuracy
# 在之前2个epoch的基础上过一遍模型
correct = 0
total = 0

with torch.no_grad():

    for images, labels in train_loader:

        x = conv1(images)
        x = torch.relu(x)
        x = pool(x)

        x = conv2(x)
        x = torch.relu(x)
        x = pool2(x)

        x = torch.flatten(x, start_dim=1)

        x = fc1(x)
        x = torch.relu(x)

        output = fc2(x)

        predicted = torch.argmax(output, dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print("Training accuracy:", accuracy)
print("Training accuracy (%):", accuracy * 100)

# [8, 3, 3, 3]
#  ↑  ↑  ↑  ↑
#  |  |  |  |
#  |  |  3×3 小窗口
#  |  |
#  |  每个 filter 要看 3 个输入通道（R/G/B）
#  |
# 一共有 8 个 filters
print("Conv weight shape:", conv1.weight.shape)

# 查看第一个filter的weights，CNN的特点就是所有的filter共享一个权重
# 起始的Pytorch的weights是随机的，后续通过backpropagation计算出gradient 后通过Optimizer更新
print(conv1.weight[0])
# 查看第一个filter对应的bias
print("Filter 1 bias:", conv1.bias[0])
# pixel × weight + bias
print("First filter, first output value:", conv_output[0, 0, 0, 0])
# Manual计算验证
import torch.nn.functional as F

padded_images = F.pad(images, (1, 1, 1, 1))

patch = padded_images[0, :, 0:3, 0:3]

manual_output = (
    patch * conv1.weight[0] # 不是矩阵乘法而是27个位置分别相乘然后通过.sum()相加起来
).sum() + conv1.bias[0]

print("Manual output:", manual_output)
print("Conv2d output:", conv_output[0, 0, 0, 0])