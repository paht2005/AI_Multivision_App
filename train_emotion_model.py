# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# 1. Load FER2013 dataset
df = pd.read_csv("fer2013.csv")
df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48) / 255.0)
X = np.stack(df['pixels'].values)
y = df['emotion'].values

# 2. Custom Dataset with augmentation
class FERDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3)], p=0.5),
            transforms.RandomCrop(44, padding=2),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  
        img = np.expand_dims(img, axis=0)  

        if self.augment:
            img = np.transpose(img, (1, 2, 0))  # convert to (H, W, C)
            img = self.transform(img)  # ToPILImage expects HWC
        else:
            img = torch.tensor(img, dtype=torch.float32)

        return img, self.y[idx]


# 3. Create Dataloaders
dataset = FERDataset(X, y, augment=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

# 4. ResNet18 model (input: grayscale, output: 7 classes)
class ResNetEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)

    def forward(self, x):
        return self.model(x)

# 5. Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetEmotion().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):  # or more
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# 6. Save the model
torch.save(model.state_dict(), "emotion_resnet18.pth")
print("✅ Model saved to emotion_resnet18.pth")
