import torch
from pytorch_nndct.apis import Inspector
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import logging
import os
from model_def import get_model


imgsz = 224
batch_size = 8
target = "0x101000016010407"
device = torch.device('cpu')
path = "/your/.pth/path"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_calibration_data(num_samples=1000):
    
    
    calibration_data = []
    for i in range(num_samples):
        
        data = torch.rand([1, 1, 224, 224])
        calibration_data.append(data)
    return calibration_data

try:
    
    print("Loading model...")
    model = get_model(num_classes=14)
    #ckpt = torch.load(path, map_location=device)
    #model.load_state_dict(ckpt)
    #model.eval()
    #model.to(device)
    ckpt = torch.load(path, map_location=device)

   
    model_dict = model.state_dict()
    matched_dict = {
        k: v for k, v in ckpt.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    # Update and load
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

    model.eval()
    model.to(device)
    
    
    print("Model loaded successfully")

    
    dummy_input = torch.rand([1, 1, 224, 224]).to(device)

    
    print("Inspecting model...")
    inspector = Inspector(target)
    inspector.inspect(model, dummy_input, device=device)
    print("Model inspection completed")

    
    print("Starting calibration phase...")
    calibration_data = load_calibration_data(num_samples=50)  
    
    quantizer = torch_quantizer("calib", model, (dummy_input), device=device)
    quant_model = quantizer.quant_model
    
    
    for i, calib_input in enumerate(calibration_data):
        calib_input = calib_input.to(device)
        with torch.no_grad():
            _ = quant_model(calib_input)
        if i % 10 == 0:
            print(f"Calibration progress: {i+1}/{len(calibration_data)}")
    
    
    quantizer.export_quant_config()
    print("Calibration completed and config exported")

    
    print("Starting test phase...")
    #quantizer = torch_quantizer("test", model, (dummy_input), device=device)
    quant_model = quantizer.quant_model
    
    
    with torch.no_grad():
        output = quant_model(dummy_input)
    print("Test phase completed")

    #   Exporting xmodel
    print("Exporting xmodel...")
    quantizer.export_xmodel()
    print("Xmodel export completed")

    
    xmodel_path = "/xmodel/path"
    if os.path.exists(xmodel_path):
        file_size = os.path.getsize(xmodel_path)
        print(f"Generated xmodel file size: {file_size} bytes")
        
        if file_size > 1000:  
            print("Xmodel file seems valid")
            
            # 
            print("Validating xmodel file...")
            try:
                os.system(f"vai_q_xir inspect --xmodel /xmodel/path")
                print("Xmodel validation passed")
                
                
                print("Starting compilation...")
                compile_cmd = f"vai_c_xir -x /.xmodle/path -a /.json/path -o /path/quantize_result -n the/file/you/want/to/save"
                result = os.system(compile_cmd)
                
                if result == 0:
                    print("Compilation completed successfully!")
                else:
                    print(f"Compilation failed with return code: {result}")
                    
            except Exception as e:
                print(f"Xmodel validation failed: {e}")
        else:
            print("WARNING: Xmodel file is too small, likely corrupted")
    else:
        print("ERROR: Xmodel file was not generated")

    
    print("Exporting ONNX model...")
    quantizer.export_onnx_model()
    print("ONNX export completed")

except Exception as e:
    print(f"Error during quantization: {e}")
    import traceback
    traceback.print_exc()
