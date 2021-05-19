import torch
import torch.nn as nn

class BasicBlockRS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(BasicBlockRS, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size = kernel_size,
            stride = 1, dilation = 1, padding = 1
        )

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size = kernel_size,
            stride = 1, dilation = 2, padding = 2
        )

        self.shortcut = nn.Sequential()
        self.activn = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.activn(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        #out = self.activn(out)

        return out

class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, kernel_size = 3):
        super(ResidualStack, self).__init__()

        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1 
        )
        self.resblocks = self._make_layers(out_channels, num_blocks, kernel_size)
        self.maxpool = nn.MaxPool1d(2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.resblocks(out)
        out = self.maxpool(out)

        return out
    
    def _make_layers(self, out_channels, num_blocks = 2, kernel_size = 3):
        layers = []
        for _ in range(num_blocks):
            # what's the output channel of the conv1 layer ?
            layers.append(BasicBlockRS(out_channels, out_channels, kernel_size))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)

class ResNetSignals(nn.Module):
    def __init__(self, in_channels, nclasses, config):
        super(ResNetSignals, self).__init__()
        self.in_channels = in_channels
        self.layer1 = ResidualStack(in_channels, 32, config[0])
        self.layer2 = ResidualStack(32, 32, config[1])
        self.layer3 = ResidualStack(32, 32, config[2])
        self.avgpool = nn.AdaptiveAvgPool1d(4)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, nclasses)
        self.activn = nn.SELU()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = self.flat(out)
        out = self.activn(self.fc(out))
        out = self.classifier(out)
    
        return out

if __name__ == '__main__':
    from torchsummary import summary
    model = ResNetSignal(2, 5, [2, 2, 2])
    model.cuda()
    summary(model, (2, 128))
    

