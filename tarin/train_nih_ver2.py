#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torchvision.models.densenet import _DenseBlock, _Transition, DenseNet

# === Config ===
CSV_PATH = "/your/CSV/path"
IMG_ROOT = "/your/image/path"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
USE_GPU = torch.cuda.is_available()

# === 14 NIH label classes ===
ih_labels = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
label_map = {label: idx for idx, label in enumerate(ih_labels)}

# === Dataset class ===
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image(self, img_name):
        # search images_001 to images_012/images 
        for i in range(1, 13):
            subdir = f"images_{i:03d}/images"
            path = os.path.join(self.root_dir, subdir, img_name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {img_name} not found in any subfolder.")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._find_image(row['Image Index'])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        labels = torch.zeros(len(label_map))
        for lbl in row['Finding Labels'].split('|'):
            if lbl in label_map:
                labels[label_map[lbl]] = 1.0
        return image, labels

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),    # resize
    transforms.ToTensor(),            # [0,1]
    transforms.Normalize([0.5], [0.5])# [-1,1]
])

# === Load and split dataset ===
dataset = NIHChestXrayDataset(CSV_PATH, IMG_ROOT, transform)
print(f"Total images from CSV: {len(dataset)}")
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,    # depends on CPU core，nomoral 2~8
    pin_memory=True   # Speeding up cuda() copies
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

#train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# === Model definition ===#
#model = models.densenet121(pretrained=False)
model = DenseNet(
    growth_rate=32,
    block_config=(6, 8, 8, 8),  # DenseNet68 layer
    num_init_features=64,
    bn_size=4,
    drop_rate=0,
    num_classes=len(label_map)
)
# Modify the first convolution layer to a single channel
model.features.conv0 = nn.Conv2d(
    in_channels=1,
    out_channels=model.features.conv0.out_channels,
    kernel_size=model.features.conv0.kernel_size,
    stride=model.features.conv0.stride,
    padding=model.features.conv0.padding,
    bias=False
)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, len(label_map)),
    nn.Sigmoid()
)
if USE_GPU:
    model = model.cuda()
    print('Using GPU')

# === Loss & optimizer ===
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Evaluation function ===
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            if USE_GPU:
                imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs).cpu().numpy()
            all_preds.append(outputs)
            all_labels.append(labels.cpu().numpy())
    preds = np.vstack(all_preds)
    labs = np.vstack(all_labels)
    aucs = []
    for i in range(labs.shape[1]):
        try:
            aucs.append(roc_auc_score(labs[:, i], preds[:, i]))
        except ValueError:
            aucs.append(float('nan'))
    return np.nanmean(aucs), aucs

# === Training loop ===
best_auc = 0.0
best_state = None
train_losses, val_aucs = [], []
torch.backends.cudnn.benchmark = True

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        if USE_GPU:
            #imgs, labels = imgs.cuda(), labels.cuda()
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    val_auc, per_class = evaluate(model, val_loader)
    val_aucs.append(val_auc)
    print(f"Epoch {epoch}: Loss={train_losses[-1]:.4f}, Val AUC={val_auc:.4f}")
    if val_auc > best_auc:
        best_auc = val_auc
        best_state = model.state_dict()

# === Save best model ===
torch.save(best_state, 'densenet68_nih14.pth')
print(f"Saved best model with Val AUC={best_auc:.4f} as densenet68_nih14.pth")

# === Plot metrics ===
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_aucs, label='Val AUC')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('training_metrics.png')
print('Saved training_metrics.png')

