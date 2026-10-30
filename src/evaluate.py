import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from config import NUM_CLASSES, DEVICE, BEST_MODEL_PATH, VAL_DIR, IMAGE_SIZE
from model import CarClassifier

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Загружаем проверочную выборку
val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

# Загружаем сохранённую модель
model = CarClassifier(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True)
)

model.eval()

all_predictions = []
all_labels = []

# Получаем предсказания
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

# Выводим метрики
print("Классы:", val_dataset.classes)

print("\nОтчёт классификации:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=val_dataset.classes,
    zero_division=0
))

print("Матрица ошибок:")
print(confusion_matrix(
    all_labels,
    all_predictions
))
