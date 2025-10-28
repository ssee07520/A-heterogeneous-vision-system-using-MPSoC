import torch
from pytorch_nndct.apis import Inspector
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import logging
import os

from model_def import get_model


imgsz = 224
batch_size = 8



target = "0x101000016010407"
logger = logging.getLogger()
device = torch.device('cpu')
path = "/your/.pth/path"
model = get_model(num_classes=14)
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt)
#optimize = 1
model.eval()
model.to(device)

in_tensor = torch.rand([1, 1, 224, 224])
in_tensor.to(device)
inspector = Inspector(target)
inspector.inspect(model, in_tensor, device=device)

in_tensor = torch.rand([1, 1, 224, 224])
in_tensor.to(device)
quantizer = torch_quantizer("calib", model, (in_tensor), device=device)
quant_model = quantizer.quant_model
quant_model(in_tensor)
quantizer.export_quant_config()


in_tensor = torch.rand([1, 1, 224, 224])
in_tensor.to(device)
quantizer = torch_quantizer("test", model, (in_tensor), device=device)
quant_model = quantizer.quant_model
quant_model(in_tensor)
quantizer.export_xmodel()
quantizer.export_onnx_model()



os.system("vai_c_xir -x /.xmodle/path -a /.json/path -o /path/quantize_result -n the/file/you/want/to/save")



#os.system("xir svg /workspace/vitis_ai_project/model/data_save/quantize_result/DenseNet_int.xmodel out.svg")
