#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pytorch_nndct.apis import torch_quantizer
#import torchxrayvision as xrv
#from inspect_float.DenseNet import DenseNet
from model_def import get_model

class XrayDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img


def run_model(model, data_loader, device, subset_len=None):
    
    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if subset_len is not None and idx >= subset_len:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            print(f"Batch {idx+1} processed")


def main():
    parser = argparse.ArgumentParser(description="DenseNet Quantization Script")
    parser.add_argument(
        "--quant_mode",
        choices=["calib", "quant", "test"],
        required=True,
        help="calib: collect calibration statistics; quant: apply quantization parameters; test: export xmodel"
    )
    parser.add_argument(
        "--subset_len",
        type=int,
        default=None,
        help="Maximum number of batches for calibration/quantization forward passes; valid for calib and quant modes only"
    )
    parser.add_argument(
        "--calib_data_dir",
        type=str,
        default="/images/path",
        help="Directory containing calibration images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="quantize_result",
        help="Output directory for quantization results"
    )
    args = parser.parse_args()




    model = get_model(num_classes=14)
    #model.load_state_dict(torch.load('densenet121_nih14_best.pth', map_location='cpu'))
    checkpoint = torch.load('densenet121_nih14_best.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # calib
    calib_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),            # -> [0,1]
        transforms.Normalize([0.5], [0.5])  # -> [-1,1]
    ])
    calib_dataset = XrayDataset(args.calib_data_dir, transform=calib_transform)
    calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 1, 224, 224).to(device)

    
    os.makedirs(args.output_dir, exist_ok=True)

    
    model = model.to(device)

    if args.quant_mode == 'calib':
        print("=== Phase 1: Calibration ===")
        dummy_input = torch.randn(8, 1, 224, 224).to(device)
        quantizer = torch_quantizer(
            quant_mode='calib',
            module=model,
            input_args=(dummy_input,)
        )
        # run_model(quantizer.quant_model, calib_loader, device, subset_len=args.subset_len)
        quant_model = quantizer.quant_model
        quant_model(dummy_input)
        quantizer.export_quant_config()
        print("Calibration finished and config exported to", args.output_dir)

    elif args.quant_mode == 'quant':
        print("=== Phase 2: Quantization ===")
        quantizer = torch_quantizer(
            quant_mode='quant',
            module=model,
            input_args=(dummy_input,),
        )
        run_model(quantizer.quant_model, calib_loader, device, subset_len=args.subset_len)
        quantizer.export_quant_config()
        print("Quantization finished and config exported to", args.output_dir)

    elif args.quant_mode == 'test':
        print("=== Phase 3: Test & Export xmodel ===")
        quantizer = torch_quantizer(
            quant_mode='test',
            module=model,
            input_args=(dummy_input,),
            device = device
        )
        
        with torch.no_grad():
            out = quantizer.quant_model(dummy_input)
            print(f"Test output shape: {out.shape}")
        quantizer.export_xmodel()
        quantizer.export_onnx_model()
        print("xmodel exported successfully to", args.output_dir)

    else:
        raise ValueError(f"Unknown quant_mode: {args.quant_mode}")


if __name__ == "__main__":
    main()

