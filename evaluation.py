
# evaluation.py
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
# models/__init__.py
from models.model import LeNet5, AlexNet, VGGNet, ResNet, GoogLeNet, Xception, SENet


# Transformation for test dataset (same as training transformation)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # If needed for color-based models
    transforms.Resize((224, 224)),  # Resize for models like VGG, ResNet, etc.
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for pretrained models
])

# Load the test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# List of models to evaluate
models = [
    LeNet5(),
    AlexNet(),
    VGGNet(),
    ResNet(),
    GoogLeNet(),
    Xception(),
    SENet()
]

# Evaluate each model
for model in models:
    model_name = model.__class__.__name__  # Get model class name as identifier
    print(f"Evaluating {model_name}...")

    # Load the trained model (replace 'model.pth' with the correct path if needed)
    model.load_state_dict(torch.load(f'{model_name.lower()}.pth'))
    model.eval()  # Set to evaluation mode

    # Initialize variables to calculate performance metrics
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Disable gradient calculation (since we are in evaluation mode)
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on test dataset ({model_name}): {accuracy}%')

    # Calculate more metrics like precision, recall, and F1-score
    print(classification_report(all_labels, all_preds))
    print("=" * 50)  # Separator between model evaluations
