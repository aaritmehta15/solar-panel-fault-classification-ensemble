import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
DATA_DIR = "./data"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(15, shear=10),
    transforms.RandomPerspective(0.2),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomAdjustSharpness(2),
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
# DATASET SPLIT
# =========================
dataset = datasets.ImageFolder(DATA_DIR)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform
test_ds.dataset.transform = val_transform

NUM_CLASSES = len(dataset.classes)

# =========================
# CLASS BALANCING
# =========================
targets = [label for _, label in train_ds]
class_counts = np.bincount(targets)

weights = 1.0 / class_counts
weights[np.argmin(class_counts)] *= 3  # boost weakest class

sample_weights = [weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# =========================
# CUSTOM CNN
# =========================
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,NUM_CLASSES)
        )

    def forward(self,x):
        return self.classifier(self.features(x))

# =========================
# MODEL LOADER
# =========================
def get_model(name):
    if name == "resnet":
        model = models.resnet50(weights="IMAGENET1K_V1")
        for param in model.parameters(): param.requires_grad = False
        for param in model.layer4.parameters(): param.requires_grad = True
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,NUM_CLASSES)
        )

    elif name == "efficientnet":
        model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        for param in model.parameters(): param.requires_grad = False
        for param in model.features[-2:].parameters(): param.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    elif name == "mobilenet":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        for param in model.parameters(): param.requires_grad = False
        for param in model.features[-1:].parameters(): param.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    elif name == "densenet":
        model = models.densenet121(weights="IMAGENET1K_V1")
        for param in model.parameters(): param.requires_grad = False
        for param in model.features[-1:].parameters(): param.requires_grad = True
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    elif name == "custom":
        model = CustomCNN()

    return model.to(DEVICE)

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, name):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_acc = 0
    patience, counter = 4, 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs,1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # VALIDATION
        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs,1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"{name} Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), f"{name}.pth")
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(torch.load(f"{name}.pth"))

    return model, train_losses, val_losses, train_accs, val_accs

# =========================
# TRAIN ALL MODELS
# =========================
model_names = ["custom","resnet","efficientnet","mobilenet","densenet"]

trained_models = {}
histories = {}

for name in model_names:
    print(f"\nTraining {name}")
    model, tl, vl, ta, va = train_model(get_model(name), name)

    trained_models[name] = model
    histories[name] = {
        "train_loss": tl,
        "val_loss": vl,
        "train_acc": ta,
        "val_acc": va
    }

# =========================
# PLOT GRAPHS
# =========================
for name, hist in histories.items():
    epochs = range(1, len(hist["train_loss"]) + 1)

    # LOSS
    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="Train Loss")
    plt.plot(epochs, hist["val_loss"], label="Val Loss")
    plt.title(f"{name} Loss")
    plt.legend()
    plt.show()

    # ACCURACY
    plt.figure()
    plt.plot(epochs, hist["train_acc"], label="Train Acc")
    plt.plot(epochs, hist["val_acc"], label="Val Acc")
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.show()

# =========================
# ENSEMBLE
# =========================
def ensemble_predict(models):
    y_true, y_pred = [], []

    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)

        outputs = []
        for name, model in models.items():
            out = torch.softmax(model(imgs), dim=1)

            if name=="efficientnet": out *= 0.4
            elif name=="resnet": out *= 0.25
            elif name=="mobilenet": out *= 0.15
            elif name=="densenet": out *= 0.1
            else: out *= 0.1

            outputs.append(out)

        final = torch.sum(torch.stack(outputs), dim=0)
        preds = torch.argmax(final,1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred

# =========================
# FINAL RESULTS
# =========================
y_true, y_pred = ensemble_predict(trained_models)

print("\nFINAL ENSEMBLE RESULTS")
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))