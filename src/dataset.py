from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import IMAGE_SIZE, TRAIN_DIR, VAL_DIR, BATCH_SIZE

# Преобразования для изображений
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Загрузка датасетов
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

# Создание загрузчиков
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Классы:", train_dataset.classes)
print("Обучающих изображений:", len(train_dataset))
print("Проверочных изображений:", len(val_dataset))

images, labels = next(iter(train_loader))

print("Размер пакета изображений:", images.shape)
print("Размер пакета меток:", labels.shape)
