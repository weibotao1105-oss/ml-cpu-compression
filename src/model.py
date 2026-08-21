import torch
import torch.nn as nn

class SmallCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.fc1 = nn.Linear(
            16*8*8,
            64
        )

        self.fc2 = nn.Linear(
            64,
            10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = SmallCNN()

    fake_images = torch.randn(4, 3, 32, 32)

    output = model(fake_images)

    print("Input shape:", fake_images.shape)
    print("Output shape:", output.shape)
