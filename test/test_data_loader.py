

from utils.data_loader import get_data_loaders



def test_data_loader():
    datasets_to_test = ['MNIST', 'FMNIST', 'CIFAR-10']
    batch_size = 64

    for dataset in datasets_to_test:
        print(f"\nTesting data loader for {dataset}")
        train_loader, test_loader = get_data_loaders(dataset_name=dataset, batch_size=batch_size)

        # Test the first batch
        data_iter = iter(train_loader)
        images, labels = next(data_iter)

        print(f"Dataset: {dataset}")
        print(f"Batch size: {images.size(0)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"First batch labels: {labels.tolist()}")  # Convert tensor to list for readability


if __name__ == "__main__":
    test_data_loader()
