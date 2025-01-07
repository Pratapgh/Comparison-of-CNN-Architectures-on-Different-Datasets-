
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(dataset_name, batch_size, root='./data'):
    """
    Returns the data loaders for the specified dataset.

    Parameters:
    - dataset_name: str, name of the dataset ('MNIST', 'FMNIST', or 'CIFAR-10')
    - batch_size: int, number of samples per batch
    - root: str, root directory to store datasets

    Returns:
    - train_loader: DataLoader for training data
    - test_loader: DataLoader for testing data
    """
    transform_mnist_fmnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
    ])

    if dataset_name == 'MNIST':
        dataset_class = datasets.MNIST
        transform = transform_mnist_fmnist
    elif dataset_name == 'FMNIST':
        dataset_class = datasets.FashionMNIST
        transform = transform_mnist_fmnist
    elif dataset_name == 'CIFAR-10':
        dataset_class = datasets.CIFAR10
        transform = transform_cifar10
    else:
        raise ValueError("Unsupported dataset. Choose from 'MNIST', 'FMNIST', or 'CIFAR-10'.")


    train_dataset = dataset_class(root=root, train=True, download=True, transform=transform)
    test_dataset = dataset_class(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
