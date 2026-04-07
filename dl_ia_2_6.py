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
# TRANSFORMS
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
# MODEL 1 (EfficientNet)
# =========================
model1 = models.efficientnet_b0(weights="IMAGENET1K_V1")

for param in model1.features[:3].parameters():
    param.requires_grad = False
for param in model1.features[3:].parameters():
    param.requires_grad = True

model1.classifier = nn.Sequential(
    nn.Linear(model1.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES)
)

model1 = model1.to(DEVICE)

# =========================
# MODEL 2 (ResNet18)
# =========================
model2 = models.resnet18(weights="IMAGENET1K_V1")

for param in model2.parameters():
    param.requires_grad = False

for param in model2.layer4.parameters():
    param.requires_grad = True

model2.fc = nn.Sequential(
    nn.Linear(model2.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, NUM_CLASSES)
)

model2 = model2.to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

optimizer1 = optim.AdamW(model1.parameters(), lr=3e-5, weight_decay=1e-4)
optimizer2 = optim.AdamW(model2.parameters(), lr=3e-5, weight_decay=1e-4)

scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=10)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, optimizer, scheduler, name):

    best_acc = 0
    patience = 6
    counter = 0

    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

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

        # VALIDATION
        model.eval()
        correct, total = 0, 0
        val_loss = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"{name} Epoch {epoch+1}: Train={train_acc:.3f} Val={val_acc:.3f}")

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), f"{name}.pth")
        else:
            counter += 1

        if counter >= patience:
            print(f"⛔ {name} Early stopping")
            break

    return train_acc_list, val_acc_list, train_loss_list, val_loss_list

# =========================
# TRAIN BOTH MODELS
# =========================
hist1 = train_model(model1, optimizer1, scheduler1, "efficientnet")
hist2 = train_model(model2, optimizer2, scheduler2, "resnet")

# =========================
# LOAD BEST MODELS
# =========================
model1.load_state_dict(torch.load("efficientnet.pth"))
model2.load_state_dict(torch.load("resnet.pth"))

model1.eval()
model2.eval()

# =========================
# TTA FUNCTION
# =========================
def tta(model, imgs):
    outputs = []
    for _ in range(3):
        aug = imgs + torch.randn_like(imgs)*0.01
        outputs.append(torch.softmax(model(aug), dim=1))
    return torch.mean(torch.stack(outputs), dim=0)

# =========================
# FINAL ENSEMBLE TEST
# =========================
correct, total = 0, 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        out1 = tta(model1, imgs)
        out2 = tta(model2, imgs)

        final = 0.7*out1 + 0.3*out2

        preds = torch.argmax(final, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

final_acc = correct / total
print("\n🔥 FINAL ENSEMBLE ACCURACY:", round(final_acc*100,2), "%")

# =========================
# PLOT GRAPHS (EfficientNet)
# =========================
epochs_range = range(1, len(hist1[0])+1)

plt.figure()
plt.plot(epochs_range, hist1[0], label="Train Acc")
plt.plot(epochs_range, hist1[1], label="Val Acc")
plt.title("EfficientNet Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs_range, hist1[2], label="Train Loss")
plt.plot(epochs_range, hist1[3], label="Val Loss")
plt.title("EfficientNet Loss")
plt.legend()
plt.show()