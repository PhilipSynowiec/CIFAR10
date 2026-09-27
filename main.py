import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from pathlib import Path

MODEL_PATH = Path('model_weights.pth')

transforms.ColorJitter()

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=train_transform,
    download=True
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    transform=test_transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

class MultiClassCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.MaxPool2d(2, 2), # (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2), # (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), # (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, 2), # (128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        y = self.stack(x)
        return y

model = MultiClassCNN().to(device)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
optimizer = torch.optim.AdamW(model.parameters())
loss_fn = nn.CrossEntropyLoss()

def train_loop():
    size = len(train_dataset)
    model.train()
    for batch, (images, labels) in enumerate(train_loader):
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        current = batch * 64 + len(images)
        print(
            f"\rLoss: {loss.item():.6f}  [{current:3d}/{size:3d}]", end="", flush=True
        )
    print()

def test_loop():
    size = len(test_dataset)
    model.eval()
    num_batches = len(test_loader)
    correct, val_loss = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            val_loss += loss_fn(logits, labels).item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    accuracy = correct / size
    val_loss /= num_batches
    print(f"Loss: {val_loss:.6f}\nAccuracy: {accuracy:.6f}")
    return val_loss

if __name__ == "__main__":
    min_val_loss = float("inf")
    count = 0
    for epoch in range(100):
        print(f"Epoch {epoch + 1}")
        train_loop()
        val_loss = test_loop()    
        if val_loss < min_val_loss:
            count = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("saved new best model")
        else:
            count += 1
            if count >= 10: # early stopping
                break


    