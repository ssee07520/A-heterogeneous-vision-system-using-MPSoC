import torch
import glob
import os
import logging
from PIL import Image

from pytorch_nndct.apis import Inspector, torch_quantizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms      

from model_def import get_model

# -- -----------------------------------
imgsz      = 224
batch_size = 8
device     = torch.device('cpu')
target     = "0x101000016010407"
pth_path   = "/your/.pth"
calib_dir  = "/your/calib_images/" 
out_xmodel = "/your/xmodel"
# --------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# --- Dataset & DataLoader -----------------
class CalibDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.files = glob.glob(os.path.join(img_dir, "*.png")) \
                   + glob.glob(os.path.join(img_dir, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        return self.transform(img)

transform = transforms.Compose([
    transforms.Resize((imgsz, imgsz)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

calib_loader = DataLoader(
    CalibDataset(calib_dir, transform),
    batch_size=batch_size,
    shuffle=False
)
# --------------------------------------------

try:
    # 1.  PyTorch 
    model = get_model(num_classes=14).to(device)
    ckpt  = torch.load(pth_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 2. inspector
    dummy = torch.rand(1,1,imgsz,imgsz).to(device)
    Inspector(target).inspect(model, dummy, device=device)

    # 3. Calibration
    quantizer = torch_quantizer("calib", model, (dummy,), device=device)
    quant_model = quantizer.quant_model
    print(">>> Calibration start")
    for i, imgs in enumerate(calib_loader):
        with torch.no_grad():
            _ = quant_model(imgs.to(device))
        if i%10==0:
            print(f"  calib {i+1}/{len(calib_loader)}")
    quantizer.export_quant_config()
    print(">>> Calibration done")

    # 4. Test & export xmodel ( scale metadata)
    quantizer = torch_quantizer("test", model, (dummy,), device=device)
    quant_model = quantizer.quant_model
    with torch.no_grad():
        _ = quant_model(dummy)
    quantizer.export_xmodel()
    print(">>> XModel exported")

    # 5. 呼叫 compiler
    cmd = (
        f"vai_c_xir -x {out_xmodel} "
        "-a /.json/path "
        "-o /path/quantize_result -n the/file/you/want/to/save"
    )
    os.system(cmd)

except Exception as e:
    logger.exception("Quantization failed")

