import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Implementasi LeNet-5 klasik yang disesuaikan untuk CIFAR-10 (3 channel, 32x32).
    Arsitektur:
    - Conv1: 3x32x32 -> 6x28x28  (kernel 5)
    - Pool 1: 6x28x28 -> 6x14x14
    - Conv2: 6x14x14 -> 16x10x10 (kernel 5)
    - Pool 2: 16x10x10 -> 16x5x5
    - Conv3: 16x5x5 -> 120x1x1   (kernel 5)
    - FC1:  120 -> 84
    - FC2:  84 -> 10 (kelas CIFAR-10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # 3 channel karena CIFAR-10: RGB
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3x32x32 -> 6x28x28 -> 6x14x14
        x = self.pool(F.relu(self.conv1(x)))
        # 6x14x14 -> 16x10x10 -> 16x5x5
        x = self.pool(F.relu(self.conv2(x)))
        # 16x5x5 -> 120x1x1
        x = F.relu(self.conv3(x))
        # flatten: (N, 120, 1, 1) -> (N, 120)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Quick sanity check ukuran output
    model = LeNet5(num_classes=10)
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print("Output shape:", out.shape)  # seharusnya [4, 10]
