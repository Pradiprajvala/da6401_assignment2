from torch.utils.data import DataLoader
import os
from torch.nn import functional as F
import wandb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.optim as optim
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self,
                 input_channels=3,  # RGB images
                 num_classes=10,
                 # Kernel sizes for each conv layer
                 filter_sizes=(3, 3, 3, 3, 3),
                 num_filters=(32, 64, 128, 256, 512),  # Filters per conv layer
                 conv_activation=F.relu,  # Activation after conv layers
                 dense_neurons=1024,  # Neurons in the dense layer
                 dense_activation=F.relu,  # Activation after dense layer
                 input_height=224,
                 input_width=224):

        super(CNNClassifier, self).__init__()

        self.conv_activation = conv_activation
        self.dense_activation = dense_activation

        self.conv_layers = nn.ModuleList()

        in_channels = input_channels
        current_height, current_width = input_height, input_width

        # Build convolutional layers
        for i in range(5):
            out_channels = num_filters[i]
            filter_size = filter_sizes[i]
            padding = filter_size // 2  # Same padding

            conv = nn.Conv2d(in_channels, out_channels, kernel_size=filter_size,
                             stride=1, padding=padding)
            self.conv_layers.append(conv)

            current_height //= 2
            current_width //= 2
            in_channels = out_channels

        dense_input_size = num_filters[-1] * current_height * current_width

        self.dense = nn.Linear(dense_input_size, dense_neurons)
        self.output = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = self.conv_activation(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = self.dense_activation(self.dense(x))
        x = self.output(x)
        return x


# Reproducibility


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# Activation Functions
activation_functions = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "mish": F.mish
}

# Dataset


class iNaturalistDataset(Dataset):
    """Custom dataset to apply transformations."""

    def __init__(self, data_dir, transform=None):
        self.dataset = ImageFolder(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Stratified Split


def create_stratified_split(dataset, val_size=0.2, seed=42):
    """Create a stratified train-validation split."""
    labels = [dataset.dataset.targets[i] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=seed
    )
    return train_idx, val_idx

# DataLoaders


def get_dataloaders(config):
    """Create data loaders with optional augmentation."""
    # Base transform
    base_transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ]

    # Data augmentation if enabled
    if config['data_augmentation']:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        ] + base_transform[1:])
    else:
        train_transform = transforms.Compose(base_transform)

    val_transform = transforms.Compose(base_transform)

    train_dataset = iNaturalistDataset(
        '/kaggle/input/nature-12k/inaturalist_12K/train', transform=train_transform)
    train_idx, val_idx = create_stratified_split(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=SubsetRandomSampler(
        train_idx), num_workers=4, pin_memory=True)
    val_loader = DataLoader(iNaturalistDataset('/kaggle/input/nature-12k/inaturalist_12K/train', transform=val_transform),
                            batch_size=config['batch_size'], sampler=SubsetRandomSampler(val_idx), num_workers=4, pin_memory=True)
    test_loader = DataLoader(iNaturalistDataset('/kaggle/input/nature-12k/inaturalist_12K/val',
                             transform=val_transform), batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

# Model Creation


def create_model(config, device):
    """Initialize and return the CNN model."""
    if config['filter_organization'] == 'same':
        filters = [int(config['base_filters'])] * 5
    elif config['filter_organization'] == 'doubling':
        filters = [int(config['base_filters'] * (2**i)) for i in range(5)]
    elif config['filter_organization'] == 'halving':
        filters = [int(config['base_filters'] / (2**i)) for i in range(5)]
    else:
        raise ValueError(
            f"Unknown filter_organization: {config['filter_organization']}")

    print("Filter configuration:", filters)

    model = CNNClassifier(
        input_channels=3,
        num_classes=10,
        filter_sizes=(3, 3, 3, 3, 3),
        num_filters=tuple(filters),
        conv_activation=activation_functions[config['activation']],
        dense_neurons=config['dense_neurons'],
        dense_activation=activation_functions[config['activation']],
        input_height=224,
        input_width=224
    ).to(device)

    if config['batch_norm']:
        bn_layers = nn.ModuleList(
            [nn.BatchNorm2d(f).to(device) for f in filters])
        original_forward = model.forward

        def forward_with_bn(x):
            for i, conv in enumerate(model.conv_layers):
                x = conv(x)
                x = bn_layers[i](x)
                x = model.conv_activation(x)
                x = F.max_pool2d(x, 2, 2)
            x = x.view(x.size(0), -1)
            x = model.dense(x)
            x = model.dense_activation(x)
            if hasattr(model, 'dropout') and model.dropout:
                x = model.dropout(x)
            return model.output(x)

        model.forward = forward_with_bn
        model.bn_layers = bn_layers

    model.dropout = nn.Dropout(
        config['dropout']) if config['dropout'] > 0 else None
    return model

# Training Functions


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader.sampler), correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader.sampler), correct / total

# Utilities


def generate_run_name(config):
    """Create a descriptive run name for wandb."""
    return (
        f"bs{config.get('batch_size')}_lr{config.get('learning_rate'):.0e}_wd{config.get('weight_decay'):.0e}_"
        f"bf{config.get('base_filters')}_{config.get('filter_organization')}_{config.get('activation')}_"
        f"dn{config.get('dense_neurons')}_do{config.get('dropout')}_bn{int(config.get('batch_norm'))}_"
        f"aug{int(config.get('data_augmentation'))}"
    )

# Training Loop


def train():
    print("Init Wandb...")
    run = wandb.init()
    print("Wandb init success.")

    config = wandb.config
    run.name = generate_run_name(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = create_model(config, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5)

    best_val_acc = 0.0
    best_config = None

    for epoch in range(config.epochs):
        print("Epoch", epoch+1)
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print("Train Accuracy", train_acc)
        print("Val Accuracy", val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            best_config = dict(config)
            with open('best_config.json', 'w') as f:
                json.dump(best_config, f, indent=4)
            wandb.save('best_model.pth')
            wandb.save('best_config.json')

    model.load_state_dict(torch.load('best_model.pth'))
    return model


# Sweep Config
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'batch_size': {'values': [32, 64, 128]},
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
        'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-3},
        'base_filters': {'values': [16, 32, 64]},
        'filter_organization': {'values': ['same', 'doubling', 'halving']},
        'activation': {'values': ['relu', 'gelu', 'silu', 'mish']},
        'dense_neurons': {'values': [512, 1024, 2048]},
        'dropout': {'values': [0.0, 0.2, 0.3, 0.5]},
        'batch_norm': {'values': [True, False]},
        'data_augmentation': {'values': [True, False]},
        'epochs': {'value': 10}
    }
}

# Run Sweep


# Define supported activation functions
activation_functions = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "mish": F.mish
}


def load_best_model(model_path, config, device):
    """Load model architecture and weights based on sweep configuration"""

    # Create model using stored config
    model = create_model(config, device)

    # Move model to device (CPU/GPU)
    model = model.to(device)

    # Apply dropout if defined
    if config['dropout'] > 0:
        model.dropout = nn.Dropout(config['dropout'])

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    return model


def get_test_dataloader(batch_size=32):
    """Prepare test DataLoader with standard transforms"""

    # Define input preprocessing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load test/validation dataset
    test_dataset = ImageFolder('/kaggle/input/nature-12k/inaturalist_12K/val',
                               transform=test_transform)

    # Wrap dataset in DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_loader, test_dataset


def evaluate_model(model, test_loader, device):
    """Run model inference and compute accuracy on test set"""

    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Run forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Update stats
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Collect predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    test_acc = 100 * correct / total

    return test_acc, all_predictions, all_labels


def create_visualization_grid(test_dataset, predictions, true_labels, class_names, num_samples=30):
    """Visualize model predictions on a grid of test samples"""

    dataset_size = len(test_dataset)

    # Identify correctly and incorrectly predicted samples
    correct_indices = [i for i in range(dataset_size)
                       if i < len(predictions) and predictions[i] == true_labels[i]]
    incorrect_indices = [i for i in range(dataset_size)
                         if i < len(predictions) and predictions[i] != true_labels[i]]

    # Sample from both correct and incorrect predictions
    num_correct = min(20, len(correct_indices))
    num_incorrect = min(10, len(incorrect_indices))
    selected_correct = random.sample(correct_indices, num_correct)
    selected_incorrect = random.sample(incorrect_indices, num_incorrect)

    # Combine and shuffle selected samples
    selected_indices = selected_correct + selected_incorrect
    random.shuffle(selected_indices)
    selected_indices = selected_indices[:num_samples]

    # Create visualization grid
    fig = plt.figure(figsize=(15, 18))
    fig.suptitle("Model Predictions on Test Data", fontsize=20)

    # Color schemes
    correct_color = '#e6f7e6'   # Light green
    incorrect_color = '#ffebeb'  # Light red

    rows, cols = 10, 3

    for i, idx in enumerate(selected_indices):
        img, label = test_dataset[idx]
        img = img.permute(1, 2, 0).cpu().numpy()

        # Undo normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        pred = predictions[idx] if idx < len(predictions) else -1
        is_correct = pred == label
        bg_color = correct_color if is_correct else incorrect_color

        # Add subplot for current image
        ax = plt.subplot(rows, cols, i + 1)
        ax.set_facecolor(bg_color)
        plt.imshow(img)

        # Annotate prediction and ground truth
        title = f"Pred: {class_names[pred]}\nTrue: {class_names[label]}"
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('test_predictions_grid.png', bbox_inches='tight', dpi=200)

    return fig


def test_model():
    print("Init Wandb...")
    run = wandb.init()
    print("Wandb init success.")

    ENTITY = "da24m012-iit-madras"      # your wandb username or team
    PROJECT = "A2"
    SWEEP_ID = "kt1c7sqy"       # just the ID, not the full path

    api = wandb.Api()

    # Get all runs in the project
    all_runs = api.runs(f"{ENTITY}/{PROJECT}")

    # Filter runs that belong to the desired sweep
    sweep_runs = [
        run for run in all_runs if run.sweep and run.sweep.id == SWEEP_ID]
    best_run = max(
        sweep_runs, key=lambda run: run.summary.get("val_accuracy", 0.0))

    # Show details
    print(f"Best run ID: {best_run.id}")
    print(f"Name: {best_run.name}")
    print(f"Best Val Accuracy: {best_run.summary['val_accuracy']}")
    print("Config:", dict(best_run.config))
    config = dict(best_run.config)
    best_run.file("best_model.pth").download(replace=True)
    model_path = "/kaggle/working/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Config", config.get("batch_size"))
    run.name = generate_run_name(config)
    print("Loading best model...")
    model = load_best_model(model_path, config, device)

    print("Preparing test data...")
    test_loader, test_dataset = get_test_dataloader(
        batch_size=config.get("batch_size"))

    # Evaluate model on test data
    print("Evaluating model on test data...")
    test_accuracy, predictions, true_labels = evaluate_model(
        model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Get class names from dataset
    class_names = test_dataset.classes
    print("Creating visualization grid...")
    fig = create_visualization_grid(
        test_dataset, predictions, true_labels, class_names)

    print("Done! Check test_predictions_grid.png for the visualization.")


def main():
    print("Success")
    sweep_id = wandb.sweep(sweep_config, project="A2")
    print("Running Sweep")
    wandb.agent(sweep_id, train, count=15)
    test_model()


if __name__ == "__main__":
    main()
