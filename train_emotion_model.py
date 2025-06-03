# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image


DATASET_PATH = "fer2013.csv"
MODEL_OUTPUT_PATH = "emotion_resnet18.pth"
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

def load_fer2013(path):
    """Load FER2013 CSV dataset and preprocess pixels."""
    df = pd.read_csv(path)
    df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48) / 255.0)
    images = np.stack(df['pixels'].values)
    labels = df['emotion'].values
    return images, labels

class FERDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3),
            transforms.RandomCrop(44, padding=2),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.expand_dims(self.images[idx], axis=0)
        if self.augment:
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
        return image, self.labels[idx]

class EmotionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)

    def forward(self, x):
        return self.model(x)

def train(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

        # Optional: validate after each epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    images, labels = load_fer2013(DATASET_PATH)
    dataset = FERDataset(images, labels, augment=True)
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionResNet().to(device)

    train(model, train_loader, val_loader, device)

    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"Model successfully saved to {MODEL_OUTPUT_PATH}")
