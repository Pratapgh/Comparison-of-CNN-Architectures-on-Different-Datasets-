
import torch
from models.exception import Xception

def test_xception():
    model = Xception(num_classes=10)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)  # CIFAR-10 input dimensions
    output = model(input_tensor)
    assert output.shape == (1, 10), "Xception output shape mismatch!"
    print("Xception test passed!")

if __name__ == "__main__":
    test_xception()
