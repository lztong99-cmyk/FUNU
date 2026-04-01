
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes = 10, layer = 2, in_channel = 1):
        super(ConvNet, self).__init__()
        self.layer_num = layer
        self.fc_input_dim = 28
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_input_dim = self.fc_input_dim//2

        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_input_dim = self.fc_input_dim//2

        if(layer >= 3):
            self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc_input_dim = self.fc_input_dim//2

        if(layer >=4): 
            self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc_input_dim = self.fc_input_dim//2
        
        self.fc = nn.Linear((self.fc_input_dim**2) * (2**(layer+1)), num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        if(self.layer_num >= 3):
            out = self.layer3(out)
        if(self.layer_num >= 4):
            out = self.layer4(out)
        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# This CNN is for MIA (shadow model) only
class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        if(input_channel == 3):
            self.classifier = nn.Sequential(
                nn.Linear(128*6*6, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )
        elif(input_channel == 1):
            self.classifier = nn.Sequential(
                nn.Linear(128*1*1, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x