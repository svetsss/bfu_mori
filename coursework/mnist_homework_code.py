import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Устройство: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_data = datasets.MNIST("./mnist", train=True, download=True, transform=transform)
val_data = datasets.MNIST("./mnist", train=False, download=True, transform=transform)

loader_args = {"batch_size": 128, "num_workers": 0, "pin_memory": device.type == "cuda"}
train_loader = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(SEED), **loader_args)
val_loader = DataLoader(val_data, shuffle=False, **loader_args)


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.10),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.30), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = MnistCNN().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

history = {"loss": [], "accuracy": []}
best_accuracy = 0.0
best_weights = None

for epoch in range(1, 9):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    total_loss = correct = total = 0
    with torch.inference_mode():
        for images, labels in val_loader:
            logits = model(images.to(device))
            total_loss += criterion(logits, labels.to(device)).item() * labels.size(0)
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    history["loss"].append(total_loss / total)
    history["accuracy"].append(accuracy)
    if accuracy > best_accuracy:
        best_accuracy, best_weights = accuracy, deepcopy(model.state_dict())
    scheduler.step()
    print(f"Эпоха {epoch}/8: accuracy = {accuracy:.2%}")

model.load_state_dict(best_weights)
baseline_accuracy = 0.9758  # accuracy исходной модели из практики
print(f"\nИсходная accuracy: {baseline_accuracy:.2%}")
print(f"Лучшая accuracy: {best_accuracy:.2%}")

epochs = range(1, 9)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs, history["loss"], marker="o")
axes[0].set(title="Validation loss", xlabel="Эпоха", ylabel="Loss")
axes[0].grid(alpha=0.3)

axes[1].bar(["Исходная", "Улучшенная"], [baseline_accuracy, best_accuracy], color=["gray", "seagreen"])
axes[1].set(title="Сравнение accuracy", ylabel="Accuracy", ylim=(0.9, 1.0))
for index, value in enumerate([baseline_accuracy, best_accuracy]):
    axes[1].text(index, value + 0.002, f"{value:.2%}", ha="center")
plt.tight_layout()
plt.show()

images, labels = next(iter(val_loader))
with torch.inference_mode():
    predictions = model(images.to(device)).argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for image, label, prediction, axis in zip(images[:10], labels[:10], predictions[:10], axes.ravel()):
    axis.imshow(image.squeeze(), cmap="gray")
    axis.set_title(f"{label} → {prediction}", color="seagreen" if label == prediction else "crimson")
    axis.axis("off")
plt.tight_layout()
plt.show()
