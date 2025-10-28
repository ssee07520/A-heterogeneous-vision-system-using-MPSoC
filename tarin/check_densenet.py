import torch
from torchvision.models.densenet import DenseNet
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import numpy as np

# 1) Modeling (DenseNet-68 architecture)
model = DenseNet(
    growth_rate=32,
    block_config=(6, 8, 8, 8),
    num_init_features=64,
    bn_size=4,
    drop_rate=0,
    num_classes=14
)
# Grayscale 1ch input
model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# 2) Load weights (handles classifier naming differences & conv0 safety checks)
state = torch.load("/your/.pth/path",
                   map_location="cpu")

# If the weights are Sequential Linear layers: classifier.0.* → classifier.*
if 'classifier.0.weight' in state:
    state['classifier.weight'] = state.pop('classifier.0.weight')
if 'classifier.0.bias' in state:
    state['classifier.bias'] = state.pop('classifier.0.bias')

# Safety check of the number of conv0 weight channels 
if 'features.conv0.weight' in state:
    w = state['features.conv0.weight']  # [64, C, 7, 7]
    if w.shape[1] == 3 and model.features.conv0.in_channels == 1:
        
        state['features.conv0.weight'] = w.mean(dim=1, keepdim=True)

missing, unexpected = model.load_state_dict(state, strict=False)
print("after remap -> missing:", missing, "unexpected:", unexpected)

model.eval()
device = torch.device("cpu")
model.to(device)

# 3) Dataset
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

class QuickNIH(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df['Finding Labels'] = self.df['Finding Labels'].fillna('No Finding')
        self.root = root
        self.t = transform
    def _find(self, name):
        for i in range(1, 13):
            p = os.path.join(self.root, f"images_{i:03d}", "images", name)
            if os.path.exists(p): return p
        return None
    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = self._find(r['Image Index'])
        if p is None:
            raise FileNotFoundError(f"Not found: {r['Image Index']}")
        im = Image.open(p).convert('L')
        x = self.t(im) if self.t else im
        return x
    def __len__(self): return len(self.df)

ds = QuickNIH("/your/csv/path",
              "/your/image/path", tf)

subset_idx = list(range(min(32, len(ds))))
loader = DataLoader(Subset(ds, subset_idx), batch_size=8, shuffle=False)

# 4) Statistical probability distribution
with torch.no_grad():
    stats = []
    for x in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        stats.append((probs.min().item(), probs.max().item(),
                      probs.mean().item(), probs.std().item()))
print("per-batch probs [min, max, mean, std]:", stats)


# ==== mAP====
from sklearn.metrics import average_precision_score

CLASS_NAMES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
    'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis',
    'Pleural_Thickening','Hernia'
]
NAME2IDX = {c:i for i,c in enumerate(CLASS_NAMES)}

class QuickNIHWithLabels(QuickNIH):
    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = self._find(r['Image Index'])
        if p is None:
            raise FileNotFoundError(f"Not found: {r['Image Index']}")
        im = Image.open(p).convert('L')
        x = self.t(im) if self.t else im
        # encode 14-d one-hot (No Finding -> all zeros)
        y = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        parts = [s.strip() for s in str(r['Finding Labels']).split('|') if s.strip()]
        if not (len(parts)==1 and parts[0].lower()=='no finding'):
            for s in parts:
                if s in NAME2IDX: y[NAME2IDX[s]] = 1.0
        return x, torch.from_numpy(y)

def evaluate_map_probs(model, loader, head_relu=False):
    model.eval()
    all_probs, all_t = [], []
    with torch.no_grad():
        for x, t in loader:
            x = x.to(device)
            logits = model(x)
            if head_relu:
                probs = torch.sigmoid(torch.relu(logits))
            else:
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_t.append(t)
    P = torch.cat(all_probs).numpy()
    T = torch.cat(all_t).numpy()
    aps=[]
    for i in range(T.shape[1]):
        if T[:,i].sum()>0:
            aps.append(average_precision_score(T[:,i], P[:,i]))
    return float(np.mean(aps)) if aps else 0.0


val_ds = QuickNIHWithLabels(
    "/your/csv/path",
    "/your/image/path", tf
)
val_loader = DataLoader(Subset(val_ds, list(range(min(512, len(val_ds))))),
                        batch_size=8, shuffle=False, num_workers=2)

map_float = evaluate_map_probs(model, val_loader, head_relu=False)
print(f"[FLOAT] mAP={map_float:.4f}")


map_float_relu = evaluate_map_probs(model, val_loader, head_relu=True)
print(f"[FLOAT + head ReLU] mAP={map_float_relu:.4f}")

