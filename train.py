import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Import your models here
from models.lenet import LeNet5
from models.alexnet import AlexNet
from models.vggnet import VGGNet
from models.resnet import ResNet
from models.googlenet import GoogLeNet
from models.exception import Xception
from models.senet import SENet

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define transforms
def get_transform(dataset_name, model_name):
    if model_name == "LeNet5":
        if dataset_name == "CIFAR10":
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
    else:
        if dataset_name in ["MNIST", "FMNIST"]:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    return transform


# Dataset and loaders
def get_dataset_and_loaders(dataset_name, model_name):
    transform = get_transform(dataset_name, model_name)
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif dataset_name == "FMNIST":
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_loss:.4f}")

    # Plot loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

    return epoch_losses


# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute metrics
    report = classification_report(all_targets, all_predictions, target_names=[str(i) for i in range(10)])
    print("\nClassification Report:\n", report)
    return report


# Main script
if __name__ == "__main__":
    datasets_to_evaluate = ["MNIST", "FMNIST", "CIFAR10"]
    models_to_evaluate = ["LeNet5", "AlexNet", "VGGNet", "ResNet", "GoogLeNet", "Xception", "SENet"]
    results = []

    for dataset_name in datasets_to_evaluate:
        for model_name in models_to_evaluate:
            print(f"\nTraining {model_name} on {dataset_name}...")

            # Load dataset
            train_loader, test_loader = get_dataset_and_loaders(dataset_name, model_name)

            # Initialize model
            if model_name == "LeNet5":
                model = LeNet5()
            elif model_name == "AlexNet":
                model = AlexNet()
            elif model_name == "VGGNet":
                model = VGGNet()
            elif model_name == "ResNet":
                model = ResNet()
            elif model_name == "GoogLeNet":
                model = GoogLeNet()
            elif model_name == "Xception":
                model = Xception()
            elif model_name == "SENet":
                model = SENet()
            else:
                raise ValueError("Invalid model name specified.")

            model = model.to(device)

            # Loss and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Train and evaluate
            train_model(model, train_loader, criterion, optimizer, epochs, device)
            report = evaluate_model(model, test_loader, device)
            results.append({"Model": model_name, "Dataset": dataset_name, "Metrics": report})

            # Save model
            model_path = f"{model_name.lower()}_{dataset_name.lower()}_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved as {model_path}")

    # Summarize results
    print("\nAll results:")
    for result in results:
        print(f"Model: {result['Model']}, Dataset: {result['Dataset']}\n{result['Metrics']}")
