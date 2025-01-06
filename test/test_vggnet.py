
import torch
from models.vggnet import VGGNet

def test_vggnet():
    model = VGGNet(num_classes=10)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)  # CIFAR-10 input dimensions
    output = model(input_tensor)
    assert output.shape == (1, 10), "VGGNet output shape mismatch!"
    print("VGGNet test passed!")

if __name__ == "__main__":
    test_vggnet()
