import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=None, stride=1, expansion_factor=4):
        super().__init__()
        self.expansion_factor = expansion_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion_factor, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion_factor)
        self.relu = nn.ReLU()
        self.downsampling = downsampling

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsampling is not None:
            identity = self.downsampling(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels, num_classes=1000,
                 expansion_factor=4, block_input_layout=(64, 128, 256, 512)):
        super().__init__()
        self.expansion_factor = expansion_factor
        self.in_channels = block_input_layout[0]  # is used during ResBlock building.
        self.conv1 = nn.Conv2d(img_channels, block_input_layout[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(block_input_layout[0])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layers is basically a list containing the arhitecture of layers i.e [3, 4, 6, 3] for ResNet50
        # It iterates over the value in the list to create the correct number of blocks
        self.layer_1 = self._make_layers(block, block_input_layout[0], res_block_number=layers[0], stride=1)
        self.layer_2 = self._make_layers(block, block_input_layout[1], res_block_number=layers[1], stride=2)
        self.layer_3 = self._make_layers(block, block_input_layout[2], res_block_number=layers[2], stride=2)
        self.layer_4 = self._make_layers(block, block_input_layout[3], res_block_number=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block_input_layout[3] * expansion_factor, num_classes)

    def _make_layers(self, block, out_channels, res_block_number, stride):
        downsampling = None
        if stride != 1 or self.in_channels != out_channels * self.expansion_factor:
            downsampling = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion_factor, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion_factor)
            )

        layers = list()
        layers.append(block(self.in_channels, out_channels, downsampling,
                            stride, expansion_factor=self.expansion_factor))
        self.in_channels = out_channels * self.expansion_factor

        for _ in range(1, res_block_number):
            layers.append(block(self.in_channels, out_channels, expansion_factor=self.expansion_factor))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def res_net_50(img_channels=3, num_classes=1000, expansion_factor=4, block_input_layout=(64, 128, 256, 512)):
    return ResNet(ResBlock, [3, 4, 6, 3], img_channels, num_classes, expansion_factor, block_input_layout)

def res_net_101(img_channels=3, num_classes=1000, expansion_factor=4, block_input_layout=(64, 128, 256, 512)):
    return ResNet(ResBlock, [3, 4, 23, 3], img_channels, num_classes, expansion_factor, block_input_layout)

def res_net_152(img_channels=3, num_classes=1000, expansion_factor=4, block_input_layout=(64, 128, 256, 512)):
    return ResNet(ResBlock, [3, 8, 36, 3], img_channels, num_classes, expansion_factor, block_input_layout)

def test():
    net_50 = res_net_50(img_channels=3,
                        num_classes=7,
                        expansion_factor=2)

    net_101_lw = res_net_101(img_channels=3,
                             num_classes=50,
                             expansion_factor=4,
                             block_input_layout=(32, 64, 128, 256))

    net_152_v2 = res_net_152(img_channels=3,
                             num_classes=1000,
                             expansion_factor=2,
                             block_input_layout=(64,256,1024,4096))

    x = torch.rand(2, 3, 224, 224) # N, C, H, W
    y_50, y_101, y_152 = net_50(x), net_101_lw(x), net_152_v2(x)
    print(f"net_50 Expected shape: (2,7) | Actual shape: {y_50.shape}")
    print(f"net_101 Expected shape: (2,50) | Actual shape: {y_101.shape}")
    print(f"net_152 Expected shape: (2,1000) | Actual shape: {y_152.shape}")

test()