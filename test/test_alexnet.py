
import torch
from models.alexnet import AlexNet

def test_alexnet():
    model = AlexNet(num_classes=10)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)  # CIFAR-10 input dimensions
    output = model(input_tensor)
    assert output.shape == (1, 10), "AlexNet output shape mismatch!"
    print("AlexNet test passed!")

if __name__ == "__main__":
    test_alexnet()
