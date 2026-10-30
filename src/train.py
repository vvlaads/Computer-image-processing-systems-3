import torch
import torch.nn as nn
import torch.optim as optim

# from collections import Counter
# from dataset import train_dataset

from model import CarClassifier
from dataset import train_loader, val_loader

from config import NUM_CLASSES, DEVICE, LEARNING_RATE, EPOCHS, BEST_MODEL_PATH


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy


model = CarClassifier(num_classes=NUM_CLASSES).to(DEVICE)

# Задаем веса классам
# class_counts = Counter(train_dataset.targets)
#
# class_weights = torch.tensor(
#     [
#         len(train_dataset) / (NUM_CLASSES * class_counts[i])
#         for i in range(NUM_CLASSES)
#     ],
#     dtype=torch.float32
# ).to(DEVICE)
#
# criterion = nn.CrossEntropyLoss(weight=class_weights)

# print("Количество изображений по классам:", class_counts)
# print("Веса классов:", class_weights)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    train_loss, train_accuracy = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )

    val_loss, val_accuracy = validate(
        model, val_loader, criterion, DEVICE
    )

    print(
        f"Эпоха [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

print("Лучшая точность:", best_val_accuracy)
