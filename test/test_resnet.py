
import torch
from models.resnet import ResNet

def test_resnet():
    model = ResNet(num_classes=10)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)  # CIFAR-10 input dimensions
    output = model(input_tensor)
    assert output.shape == (1, 10), "ResNet output shape mismatch!"
    print("ResNet test passed!")

if __name__ == "__main__":
    test_resnet()
