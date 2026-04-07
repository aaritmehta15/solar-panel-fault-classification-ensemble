import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import Counter

# =========================
# CONFIG
# =========================
DATA_DIR = "./data"
BATCH_SIZE = 8
IMG_SIZE = 224
EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS (BALANCED)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# DATASET (NO LEAKAGE)
# =========================
dataset = datasets.ImageFolder(DATA_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform

# Data leakage check
assert len(set(train_ds.indices) & set(val_ds.indices)) == 0
print("✅ No Data Leakage")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

NUM_CLASSES = len(dataset.classes)

# =========================
# CLASS WEIGHTS
# =========================
targets = [label for _, label in train_ds]
class_counts = Counter(targets)

weights = [1.0/class_counts[i] for i in range(NUM_CLASSES)]
weights = torch.tensor(weights).to(DEVICE)

# =========================
# MODEL (BALANCED)
# =========================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Freeze few early layers
for param in model.features[:3].parameters():
    param.requires_grad = False

# Train rest
for param in model.features[3:].parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# =========================
# TRAINING + EARLY STOPPING
# =========================
best_acc = 0
patience = 6
counter = 0

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

for epoch in range(EPOCHS):

    model.train()
    correct, total = 0, 0
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    correct, total = 0, 0
    val_running_loss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_loss = val_running_loss / len(val_loader)

    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)

    scheduler.step()

    print(f"Epoch {epoch+1}: Train={train_acc:.3f} Val={val_acc:.3f}")

    # EARLY STOPPING
    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print("⛔ Early stopping triggered")
        break

# =========================
# LOAD BEST MODEL
# =========================
model.load_state_dict(torch.load("best_model.pth"))

# =========================
# FINAL TEST (VAL AS TEST)
# =========================
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        preds = torch.argmax(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

final_acc = correct / total

print("\n🔥 FINAL ACCURACY:", round(final_acc * 100, 2), "%")

# =========================
# PLOTS
# =========================
epochs_range = range(1, len(train_acc_list)+1)

# Accuracy Graph
plt.figure()
plt.plot(epochs_range, train_acc_list, label="Train Accuracy")
plt.plot(epochs_range, val_acc_list, label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()

# Loss Graph
plt.figure()
plt.plot(epochs_range, train_loss_list, label="Train Loss")
plt.plot(epochs_range, val_loss_list, label="Validation Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()