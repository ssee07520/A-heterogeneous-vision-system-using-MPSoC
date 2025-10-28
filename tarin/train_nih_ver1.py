import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchxrayvision as xrv
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === Config ===
IMG_DIR_PATH = "/your/image/path"
CSV_PATH = "//your/CSV/path"
BATCH_SIZE = 32
NUM_EPOCHS = 15
USE_GPU = torch.cuda.is_available()

# === 14 NIH label classes ===
nih_labels = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
label_map = {label: idx for idx, label in enumerate(nih_labels)}

# === Dataset class ===
class NIHChestXrayFolderDataset(Dataset):
    def __init__(self, root_dir, csv_path, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = nih_labels  

    def _parse_label(self, label_str):
        label_vec = torch.zeros(len(self.label_map))
        for label in label_str.split("|"):
            if label in self.label_map:
                label_vec[self.label_map.index(label)] = 1.0
        return label_vec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]

        for i in range(1, 13):
            img_path = os.path.join(self.root_dir, f"images_{i:03d}", "images", img_name)
            if os.path.exists(img_path):
                break
        else:
            print(f"Image {img_name} not found. Skipping.")
            return self.__getitem__((idx + 1) % len(self.df))

        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
            print(f"[DEBUG] index {idx} image shape: {image.shape}")
#            print(f"image min: {image.min().item():.2f}, max: {image.max().item():.2f}, dtype: {image.dtype}")

        label = self._parse_label(row["Finding Labels"])
        return image, label

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] assuming original range [0, 255]
])



# === Load dataset and split ===
full_dataset = NIHChestXrayFolderDataset(root_dir=IMG_DIR_PATH, csv_path=CSV_PATH, transform=transform)
train_indices, val_indices = train_test_split(list(range(len(full_dataset))), test_size=0.1, random_state=42)
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# === Model ===
model = xrv.models.DenseNet(weights="densenet121-res224-nih")
model.op_threshs = None
model.classifier = nn.Sequential(
    nn.Linear(1024, len(label_map)),
    nn.Sigmoid()
)
if USE_GPU:
    print("Using GPU")
    model = model.cuda()

# === Loss and optimizer ===
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Evaluation function ===
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            if USE_GPU:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# === Training Loop ===
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        if USE_GPU:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = evaluate(model, val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()

# === Save model ===
torch.save(best_model_state, "densenet121_nih14_best.pth")
print("Done densenet121_nih14_best.pth")



model.eval()  # ??????
all_outputs = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:  # val_loader ??????
        outputs = model(images)        # outputs.shape: [batch_size, num_classes]
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

all_outputs = torch.cat(all_outputs).numpy()  # [num_samples, num_classes]
all_labels = torch.cat(all_labels).numpy()    # [num_samples, num_classes]
print(all_outputs.shape, all_labels.shape)



# === Plot loss curve ===
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.savefig("loss_curve.png")
print("Print loss_curve.png")
