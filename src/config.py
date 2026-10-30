import random
import torch
import numpy as np
from pathlib import Path

# Сид для повторного запуска
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Пути проекта
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = PROJECT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model_augmented.pth"

# Параметры изображений
IMAGE_SIZE = 128
BATCH_SIZE = 16

# Параметры обучения
NUM_CLASSES = 3
EPOCHS = 15
LEARNING_RATE = 0.001

# Вычислительное устройство
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Создание папки моделей, если её ещё нет
MODELS_DIR.mkdir(parents=True, exist_ok=True)
