# Imports
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall

# Load datasets
from torchvision import datasets
import torchvision.transforms as transforms

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Define train and test dataloaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Find the number of classes and image size
num_classes = len(train_data.classes)
image_size = train_data[0][0].shape[1]
input_channels = 1
output_channels = 16


# Define simple CNN classifier model
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(output_channels * (image_size // 2) ** 2, num_classes)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.flatten(x)
        # x = x.view(-1, output_channels * (image_size//2)**2)
        x = self.fc(x)
        return x


model = CNNClassifier(num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'epoch: {epoch}, loss: {loss}')

# Testing and obtaining predictions
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted)
        true_labels.extend(labels)

# Define metrics
accuracy_metric = Accuracy(task="multiclass", num_classes=10)
precision_metric = Precision(task="multiclass", num_classes=10, average=None)
recall_metric = Recall(task="multiclass", num_classes=10, average=None)

# Convert predictions and labels to tensors for metrics
predictions_tensor = torch.tensor(predictions)
true_labels_tensor = torch.tensor(true_labels)

# Calculate metrics
accuracy = accuracy_metric(predictions_tensor, true_labels_tensor).item()
precision = precision_metric(predictions_tensor, true_labels_tensor).tolist()
recall = recall_metric(predictions_tensor, true_labels_tensor).tolist()
print(f"Accuracy: {accuracy}")
print(f"Precision (per class): {precision}")
print(f"Recall (per class): {recall}")
