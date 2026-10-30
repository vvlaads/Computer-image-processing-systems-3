import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model import CarClassifier
from config import (
    DEVICE,
    NUM_CLASSES,
    BEST_MODEL_PATH,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CarClassifier(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Классы:", test_dataset.classes)
print("Количество тестовых изображений:", len(test_dataset))

print("\nAccuracy:", accuracy_score(all_labels, all_predictions))

print("\nОтчёт классификации:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes,
    zero_division=0
))

print("Матрица ошибок:")
print(confusion_matrix(all_labels, all_predictions))
