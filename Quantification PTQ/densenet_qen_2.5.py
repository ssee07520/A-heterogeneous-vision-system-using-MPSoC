from pytorch_nndct.apis import torch_quantizer
import torch
import torchxrayvision as xrv
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# 1. Prepare the model
model = xrv.models.DenseNet(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(1024, 14),
    nn.Sigmoid()
)
model.load_state_dict(torch.load('densenet68_nih14.pth', map_location='cpu'))
model.eval()

# 2. Preparing the calibration data set
class XrayDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                        if f.endswith('.png') or f.endswith('.jpg')]
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img

calib_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
#    transforms.Lambda(lambda x: x * 2048 - 1024)
])

calib_dataset = XrayDataset('/images/path', transform=calib_transform)
calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False)

dummy_input = torch.randn(1, 1, 224, 224)

# Make sure the output directory exists
os.makedirs('quantize_result', exist_ok=True)

def run_model(model, data_loader, device):
    """Run the model for calibration"""
    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            print(f"Calibration batch {idx+1} done")

# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print("Starting quantization process...")

# step1:Calibration
print("\n=== Phase 1: Calibration ===")
quantizer = torch_quantizer(
    quant_mode='calib',
    module=model,
    input_args=(dummy_input.to(device),),
    output_dir='quantize_result'
)


quantized_model = quantizer.quant_model
quantized_model.eval()


run_model(quantized_model, calib_loader, device)

quantizer.export_quant_config()
print("Calibration finished and config exported")

# step2:Quantization
print("\n=== Phase 2: Quantization ===")
quantizer_quant = torch_quantizer(
    quant_mode='quant',
    module=model,
    input_args=(dummy_input.to(device),),
    output_dir='quantize_result'
)


quantized_model_quant = quantizer_quant.quant_model
quantized_model_quant.eval()


run_model(quantized_model_quant, calib_loader, device)

quantizer_quant.export_quant_config()
print("Quantization finished and config exported")

# step3:Test
print("\n=== Phase 3: Test ===")
quantizer_test = torch_quantizer(
    quant_mode='test',
    module=model,
    input_args=(dummy_input.to(device),),
    output_dir='quantize_result'
)


quantized_model_test = quantizer_test.quant_model
quantized_model_test.eval()


with torch.no_grad():
    output = quantized_model_test(dummy_input.to(device))
    print(f"Test output shape: {output.shape}")


quantizer_test.export_xmodel(output_dir='quantize_result')
print("xmodel exported successfully")


folder = '/images/path'
print(f"Total calibration images: {len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))])}")


output_files = os.listdir('quantize_result')
print(f"Output files: {output_files}")


json_files = [f for f in output_files if f.endswith('.json')]
if json_files:
    print(f"JSON files generated: {json_files}")
else:
    print("No JSON files found in output directory")
