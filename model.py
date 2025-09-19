import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        
        # Optimized feature extraction - much smaller channels
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.0),
            
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(6, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.0),
            
            # Block 3: 7x7 -> 3x3
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.0),

             #Block 4: 3x3 -> 1x1
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.0),
        )
        
        # Minimal classifier (features end at 52 x 1 x 1 after Block 4 pooling)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 1 * 1, num_classes)
        )
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


    
    
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# def create_optimized_model(target_params=20000):
#     """Create a model with parameters close to target_params"""
#     model = Model()
#     total, trainable = count_parameters(model)
    
#     print(f"Initial model: {total} parameters")
    
#     if total <= target_params:
#         return model
    
#     # If still too large, create an even smaller model
#     class TinyModel(nn.Module):
#         def __init__(self, num_classes=10):
#             super(TinyModel, self).__init__()
#             self.features = nn.Sequential(
#                 nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 28x28 -> 28x28
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
#                 nn.Conv2d(6, 12, kernel_size=5, padding=2),  # 14x14 -> 14x14
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
#             )
#             self.classifier = nn.Linear(12 * 7 * 7, num_classes)
        
#         def configure_optimizers(self):
#             return optim.Adam(self.parameters(), lr=0.001)
        
#         def forward(self, x):
#             x = self.features(x)
#             x = x.view(x.size(0), -1)
#             x = self.classifier(x)
#             return F.log_softmax(x, dim=1)
    
#     tiny_model = TinyModel()
#     total, trainable = count_parameters(tiny_model)
#     print(f"Tiny model: {total} parameters")
    
#     return tiny_model


def evaluate(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_examples += target.size(0)
    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return avg_loss, accuracy


def train(model, train_loader, val_loader, epochs):
    optimizer = model.configure_optimizers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []  # per-epoch average
    val_losses = []    # per-epoch average
    train_batch_losses = []  # per-batch loss
    val_batch_losses = []    # per-batch validation loss
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        val_iter = iter(val_loader)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * target.size(0)
            preds = output.argmax(dim=1)
            running_correct += (preds == target).sum().item()
            running_examples += target.size(0)

            # Record and print per-batch training loss
            train_batch_losses.append(loss.item())
            # Compute per-batch validation loss (one batch from val loader)
            model.eval()
            with torch.no_grad():
                try:
                    val_data, val_target = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_data, val_target = next(val_iter)
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                val_loss_batch = F.nll_loss(val_output, val_target).item()
                val_batch_losses.append(val_loss_batch)
            model.train()

         

        epoch_train_loss = running_loss / max(running_examples, 1)
        epoch_train_acc = running_correct / max(running_examples, 1)

        val_loss, val_acc = evaluate(model, val_loader)

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_batch_losses': train_batch_losses,
        'val_batch_losses': val_batch_losses,
    }
    
def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


def write_metrics_to_readme(history, test_acc):
    """Write training metrics to README.md"""
    readme_path = 'README.md'
    
    # Build metrics table
    table_lines = [
        "### Training Metrics (per epoch)",
        "",
        "| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |",
        "|------:|-----------:|----------:|---------:|--------:|"
    ]
    
    for i, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(
        history['train_losses'], history['train_accuracies'],
        history['val_losses'], history['val_accuracies']
    ), 1):
        table_lines.append(
            f"| {i} | {train_loss:.4f} | {train_acc:.4f} | {val_loss:.4f} | {val_acc:.4f} |"
        )
    
    table_lines.extend([
        "",
        f"**Final Test Accuracy: {test_acc:.4f}**",
        "",
        "*Metrics updated automatically after training.*"
    ])
    
    # Read existing README
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Find and replace the metrics section
        start_marker = "### Training Metrics (per epoch)"
        end_marker = "*Metrics updated automatically after training.*"
        
        start_idx = content.find(start_marker)
        if start_idx != -1:
            # Find the end of the metrics section
            end_idx = content.find(end_marker, start_idx)
            if end_idx != -1:
                end_idx = content.find('\n', end_idx) + 1
                # Replace the section
                new_content = content[:start_idx] + '\n'.join(table_lines) + content[end_idx:]
            else:
                # Append if end marker not found
                new_content = content + '\n\n' + '\n'.join(table_lines)
        else:
            # Append if start marker not found
            new_content = content + '\n\n' + '\n'.join(table_lines)
    else:
        new_content = '\n'.join(table_lines)
    
    # Write back to README
    with open(readme_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Metrics written to {readme_path}")


def plot_losses(history):
    # Per-epoch plot
    if len(history['train_losses']) > 0:
        plt.figure(figsize=(9, 5))
        epochs = range(1, len(history['train_losses']) + 1)
        plt.plot(epochs, history['train_losses'], label='Train Loss (epoch)')
        plt.plot(epochs, history['val_losses'], label='Val Loss (epoch)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Per-epoch Training and Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    # model = create_optimized_model(target_params=20000)
    model = Model()
    total_params, trainable_params = count_parameters(model)
    print(f"Final model - Total parameters: {total_params} | Trainable parameters: {trainable_params}")
    
    if total_params > 20000:
        print(f"⚠️  Warning: Model has {total_params} parameters, exceeding 20K target")
    else:
        print(f"✅ Model optimized: {total_params} parameters (under 20K target)")
    model, history = train(model, train_loader, test_loader, 10)
    test_acc = test(model, test_loader)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    plot_losses(history)
    write_metrics_to_readme(history, test_acc)
