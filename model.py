import torch
import torch.nn as nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.inception1 = self._inception_block(3, 64, 128, 128, 32, 32)
        self.inception2 = self._inception_block(256, 128, 192, 96, 64)
        self.inception3 = self._inception_block(480, 192, 208, 48, 64)
        self.inception4 = self._inception_block(512, 160, 224, 64, 64)
        self.inception5 = self._inception_block(512, 128, 256, 64, 64)
        self.fc = nn.Linear(512, num_classes)

    def _inception_block(self, in_channels, out1, out2, out3, out4, out5):
        return nn.Sequential(
            nn.Conv2d(in_channels, out1, kernel_size=1),
            nn.Conv2d(out1, out2, kernel_size=3, padding=1),
            nn.Conv2d(out1, out3, kernel_size=5, padding=2),
            nn.Conv2d(in_channels, out4, kernel_size=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out4, out5, kernel_size=1)
        )

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
