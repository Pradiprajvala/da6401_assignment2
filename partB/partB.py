import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import os
import wandb
import datetime
import socket
import platform
import random
import string

# This function crafts a unique and descriptive name for each WandB run


def generate_run_name(model_name, batch_size, lr, num_classes):
    date_str = datetime.datetime.now().strftime("%m%d")
    rand_suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
    run_name = f"{model_name}-c{num_classes}-b{batch_size}-lr{lr:.1e}-{date_str}-{rand_suffix}"
    return run_name

# Handles one full training pass through the data


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()

# Runs evaluation for one epoch, without updating model weights


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()

# Full training loop across multiple epochs with logging, saving, and tracking


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, model_name, num_epochs=15):
    start_time = time.time()
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    os.makedirs('models', exist_ok=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(
                'models', f'best_model_{wandb.run.name}.pth')
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
            print(f'New best model saved with accuracy: {best_acc:.4f}')

            artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    wandb.run.summary["best_val_accuracy"] = best_acc
    wandb.run.summary["training_time"] = time_elapsed

    model.load_state_dict(torch.load(os.path.join(
        'models', f'best_model_{wandb.run.name}.pth')))

    return model, train_losses, val_losses, train_accs, val_accs

# Evaluates trained model on a test set and logs useful stats


def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    corrects = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

            c = (preds == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = corrects.double() / total
    print(f'Test Accuracy: {test_acc:.4f}')

    cm = confusion_matrix(all_labels, all_preds)

    class_accuracy = {}
    for i in range(len(class_names)):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f'Accuracy of {class_names[i]}: {100 * acc:.2f}%')
            class_accuracy[f"class_acc_{class_names[i]}"] = acc

    wandb.run.summary["test_accuracy"] = test_acc.item()
    wandb.run.summary.update(class_accuracy)

    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=class_names)
    })

    return test_acc.item(), cm

# Saves and logs plots for loss/accuracy and confusion matrix


def plot_results(train_losses, val_losses, train_accs, val_accs, cm, class_names):
    os.makedirs('plots', exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt_path = os.path.join('plots', f'training_curves_{wandb.run.name}.png')
    plt.savefig(plt_path)
    wandb.log({"training_curves": wandb.Image(plt_path)})
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join('plots', f'confusion_matrix_{wandb.run.name}.png')
    plt.savefig(cm_path)
    wandb.log({"confusion_matrix_plot": wandb.Image(cm_path)})
    plt.close()

# Prints out which model layers will be trained


def print_params_to_train(model):
    print("Layers being trained:")
    params_to_train = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_train.append(name)
    print(f"Parameters to train: {params_to_train}")

# Main script to set everything up and kick off training


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    data_dir = "/kaggle/input/nature-12k/inaturalist_12K"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    full_train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(val_dir, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    model_name = "resnet50"
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print_params_to_train(model)
    num_epochs = 15
    run_name = generate_run_name(
        model_name, batch_size, learning_rate, num_classes)

    wandb.init(
        project="A2",
        name=run_name,
        config={
            "architecture": model_name,
            "dataset": data_dir,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "scheduler_step_size": scheduler.step_size,
            "scheduler_gamma": scheduler.gamma,
            "num_classes": num_classes,
            "class_names": class_names,
            "device": device.type,
            "frozen_backbone": True,
            "system_info": f"{platform.system()} {platform.release()} - {socket.gethostname()}",
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset)
        }
    )

    wandb.watch(model, criterion, log="all", log_freq=10)

    print("Starting model training...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, model_name, num_epochs
    )

    print("Evaluating model on test set...")
    test_acc, cm = evaluate_model(model, test_loader, device, class_names)

    plot_results(train_losses, val_losses, train_accs,
                 val_accs, cm, class_names)

    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"WandB run: {wandb.run.name}")

    model_artifact = wandb.Artifact(
        f"trained-model-{wandb.run.id}",
        type="model",
        description=f"Final trained {model_name} model on naturalist dataset"
    )
    model_path = os.path.join("models", f"final_model_{wandb.run.name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': num_epochs,
        'test_accuracy': test_acc,
        'class_names': class_names
    }, model_path)
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
