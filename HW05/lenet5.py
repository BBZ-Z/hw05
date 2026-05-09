# lenet5.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 经典LeNet-5结构适配MNIST（输入28x28）
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 输出28x28x6
        self.pool1 = nn.AvgPool2d(2, 2)  # 输出14x14x6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输出10x10x16
        self.pool2 = nn.AvgPool2d(2, 2)  # 输出5x5x16
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)  # 输出1x1x120
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x