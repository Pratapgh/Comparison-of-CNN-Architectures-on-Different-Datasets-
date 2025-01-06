# model.py
import torch
import torch.nn as nn
import timm
import torchvision.models as models

# LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, 10)  # Adjust for MNIST classes

    def forward(self, x):
        return self.model(x)

# VGGNet
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, 10)  # Adjust for MNIST classes

    def forward(self, x):
        return self.model(x)

# ResNet
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)  # Adjust for MNIST classes

    def forward(self, x):
        return self.model(x)

# GoogLeNet
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.model = models.googlenet(pretrained=True)
        self.model.fc = nn.Linear(1024, 10)  # Adjust for MNIST classes

    def forward(self, x):
        return self.model(x)

# Xception

class Xception(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(Xception, self).__init__()
        # Use timm to load Xception model
        self.model = timm.create_model('xception', pretrained=pretrained)
        in_features = self.model.get_classifier().in_features  # Get input features of the classifier
        self.model.fc = nn.Linear(in_features, num_classes)  # Adjust for MNIST classes

    def forward(self, x):
        return self.model(x)



# SENet (from external library)
class SENet(nn.Module):
    def __init__(self, num_classes=10):
        super(SENet, self).__init__()
        self.model = se_resnet50(pretrained='imagenet')
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)  # Adjust for MNIST classes

    def forward(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)
