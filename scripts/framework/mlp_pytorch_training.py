import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Parameters
input_shape = (1, 28, 28)  # PyTorch uses NCHW format for images
num_classes = 10
learning_rate = 0.0002
batch_size = 32
epochs = 30
l2_lambda = 0.0001
use_gpu = True

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 45)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(45, 35)
        self.fc3 = nn.Linear(35, 23)
        self.fc4 = nn.Linear(23, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)
    
if __name__ == "__main__":

    num_works=0
    if use_gpu:
        mp.set_start_method('spawn', force=True)
        num_works=4

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to PyTorch tensor and normalizes to [0, 1]
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_works, pin_memory=use_gpu
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_works, pin_memory=use_gpu
    )

    # Initialize the model
    model = SimpleNN()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # This includes softmax internally
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Check for GPU availability
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)

    # Mixed Precision Scaler for training
    scaler = torch.cuda.amp.GradScaler() if use_gpu else None

    # Training loop
    print("Training started...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=use_gpu):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if use_gpu:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Evaluation
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Mixed precision inference
            with torch.cuda.amp.autocast(enabled=use_gpu):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    end_time = time.time()
    testing_time = end_time - start_time
    print(f"Testing time: {testing_time:.2f} seconds")

    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    print(f"Results:")
    print(f" Accuracy: {accuracy:.2f}%")
    print(f" Precision: {precision:.2f}%")
    print(f" Recall: {recall:.2f}%")
    print(f" F1-Score: {f1:.2f}%")

    # Model Configuration
    print("Neural Network Configuration:")
    print(f" Input dimension: {input_shape[1] * input_shape[2]}")
    print(f" Depth: 4")
    print(f"  Layer 0 dimension: 45, activation : relu")
    print(f"  Layer 1 dimension: 35, activation : relu")
    print(f"  Layer 2 dimension: 23, activation : relu")
    print(f"  Layer 3 dimension: {num_classes}, activation : softmax")
    print(f" Learning Rate: {learning_rate}")
    print(f" Max Epochs: {epochs}")
    print(f" Batch Size: {batch_size}")
    print(f" Optimizer: Adam")
    print(f" Error Loss: cross entropy")
    print(f" L2 Regularization Lambda: {l2_lambda}")