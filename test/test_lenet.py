

import torch
from models.lenet import LeNet5
from utils.data_loader import get_data_loaders

def test_lenet():
    # Parameters
    dataset_name = "MNIST"  # Change this to "FMNIST" or "CIFAR-10" for other datasets
    batch_size = 64
    input_channels = 1 if dataset_name in ["MNIST", "FMNIST"] else 3
    num_classes = 10

    # Load data
    train_loader, test_loader = get_data_loaders(dataset_name=dataset_name, batch_size=batch_size)

    # Initialize model
    model = LeNet5(input_channels=input_channels, num_classes=num_classes)
    model.eval()  # Set to evaluation mode

    # Get a batch of data
    images, labels = next(iter(train_loader))

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    # Print results
    print(f"Testing LeNet-5 with {dataset_name} dataset")
    print(f"Input batch shape: {images.shape}")
    print(f"Output batch shape: {outputs.shape}")
    print(f"Output (first 5 predictions): {outputs[:5]}")

if __name__ == "__main__":
    test_lenet()
