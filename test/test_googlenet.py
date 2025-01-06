import torch
from models.googlenet import GoogLeNet

def test_googlenet():
    model = GoogLeNet(num_classes=10)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)  # CIFAR-10 input dimensions
    output = model(input_tensor)
    assert output.shape == (1, 10), "GoogLeNet output shape mismatch!"
    print("GoogLeNet test passed!")

if __name__ == "__main__":
    test_googlenet()
