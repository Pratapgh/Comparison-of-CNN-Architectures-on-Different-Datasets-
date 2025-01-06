import torch
from models.senet import SENet

def test_senet():
    model = SENet(num_classes=10)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)  # CIFAR-10 input dimensions
    output = model(input_tensor)
    assert output.shape == (1, 10), "SENet output shape mismatch!"
    print("SENet test passed!")

if __name__ == "__main__":
    test_senet()
