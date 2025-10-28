#import torch.nn as nn
#from torchvision import models
#from torchvision.models.densenet import DenseNet

#def get_model(num_classes=14):
#    model = models.densenet121(pretrained=False)
#    model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels,
#                                     model.features.conv0.kernel_size,
#                                     model.features.conv0.stride,
#                                     model.features.conv0.padding,
#                                     bias=False)
#    model.classifier = nn.Sequential(
#        nn.Linear(1024, num_classes),
#        nn.ReLU()
#    )
#    return model

def get_model(num_classes=14):
   # 1. ???? DenseNet121:?? block ?? 8 ? layer
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 8, 8, 8),    # ???? (6,12,24,16) ?? (6,8,8,8)
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=num_classes,
    )
   # 2. ???????????
    model.features.conv0 = nn.Conv2d(
        1,
        model.features.conv0.out_channels,
        kernel_size=model.features.conv0.kernel_size,
        stride=model.features.conv0.stride,
        padding=model.features.conv0.padding,
        bias=False
    )

    # 3. ??????? ReLU,?????? classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, num_classes),
        nn.Sigmoid()
        nn.ReLU()
    )

    return model


# model_def_quant.py
#import torch
#import torch.nn as nn
#from typing import Tuple
#from collections import OrderedDict
#import re

#def _remap_tv_to_custom(state: dict, block_config=(6,8,8,8)) -> "OrderedDict[str, torch.Tensor]":
    """? torchvision DenseNet ???? ? ??? DenseNet68 ????"""
#    out = OrderedDict()
#    def put(k_new, v): out[k_new] = v

    # 1) stem: features.conv0 / features.norm0 ? stem.0 / stem.1
#    for k, v in list(state.items()):
#        if k.startswith("features.conv0."):
#            suffix = k.split(".", 2)[2]
            # ? conv0 ? 3ch ? 1ch
#            if suffix == "weight" and v.ndim == 4 and v.shape[1] == 3:
#                v = v.mean(dim=1, keepdim=True)
#            put(f"stem.0.{suffix}", v)
#        elif k.startswith("features.norm0."):
#            suffix = k.split(".", 2)[2]
#            put(f"stem.1.{suffix}", v)
        # relu0 / pool0 ???,?

    # 2) denseblocks / transitions
    #   features.denseblock{b}.X ? features.{2*(b-1)}.X
    #   features.transition{t}.(norm/conv) ? features.{2*t-1}.(norm/conv)
#    for k, v in list(state.items()):
#        m = re.match(r"features\.denseblock(\d+)\.(.+)", k)
#        if m:
#            b = int(m.group(1))         # 1..4
#            suffix = m.group(2)
#            idx = 2 * (b - 1)           # 0,2,4,6
#            put(f"features.{idx}.{suffix}", v)
#            continue
#        m = re.match(r"features\.transition(\d+)\.(.+)", k)
#        if m:
#            t = int(m.group(1))         # 1..3
#            suffix = m.group(2)         # norm.* / conv.weight / pool.*(???)
#            if suffix.startswith("pool"):
#                continue
#            idx = 2 * t - 1             # 1,3,5
#            put(f"features.{idx}.{suffix}", v)
#            continue

    # 3) ???? BN:features.norm5 ? features.{2*len(block_config)-1}
##    for k, v in list(state.items()):
#        m = re.match(r"features\.norm5\.(.+)", k)
#        if m:
#            final_idx = 2 * len(block_config) - 1  # (6,8,8,8) ? 7
#            put(f"features.{final_idx}.{m.group(1)}", v)

    # 4) classifier:Sequential(Linear,Sigmoid) ? Linear
#    if "classifier.0.weight" in state:
#        put("classifier.weight", state["classifier.0.weight"])
#    if "classifier.0.bias" in state:
#        put("classifier.bias", state["classifier.0.bias"])
#    if "classifier.weight" in state and "classifier.weight" not in out:
#        put("classifier.weight", state["classifier.weight"])
#    if "classifier.bias" in state and "classifier.bias" not in out:
#        put("classifier.bias", state["classifier.bias"])

#    return out

# --------- DenseNet 68(?? / DPU ???)---------
#class _DenseLayer(nn.Module):
#   def __init__(self, in_feat, growth, bn_size, drop_rate=0.0):
#        super().__init__()
#        self.norm1 = nn.BatchNorm2d(in_feat)
#        self.relu1 = nn.ReLU(inplace=False)
#        self.conv1 = nn.Conv2d(in_feat, bn_size*growth, kernel_size=1, stride=1, bias=False)
#        self.norm2 = nn.BatchNorm2d(bn_size*growth)
#        self.relu2 = nn.ReLU(inplace=False)
#        self.conv2 = nn.Conv2d(bn_size*growth, growth, kernel_size=3, stride=1, padding=1, bias=False)
#        self.drop = drop_rate

#    def forward(self, x):
#        out = self.relu1(self.norm1(x))
#        out = self.conv1(out)
#        out = self.relu2(self.norm2(out))
#        out = self.conv2(out)
#        if self.drop > 0:
#            out = nn.functional.dropout(out, p=self.drop, training=self.training)
        # DenseNet: concat ??
#        return torch.cat([x, out], dim=1)

#class _DenseBlock(nn.Sequential):
#    def __init__(self, n_layers, in_feat, bn_size, growth, drop_rate=0.0):
#        super().__init__()
 #       feat = in_feat
 #       for i in range(n_layers):
 #           layer = _DenseLayer(feat, growth, bn_size, drop_rate)
#            self.add_module(f"denselayer{i+1}", layer)
#           feat += growth

#class _Transition(nn.Module):
#    def __init__(self, in_feat, out_feat):
#        super().__init__()
#        self.norm = nn.BatchNorm2d(in_feat)
#        self.relu = nn.ReLU(inplace=False)
#        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1, bias=False)
#        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

#    def forward(self, x):
#        x = self.relu(self.norm(x))
#        x = self.conv(x)
#        x = self.pool(x)
#        return x

#class DenseNet68(nn.Module):
#    def __init__(self,
#                 in_ch: int = 1,                 # ??
#                 growth_rate: int = 32,
#                 block_config: Tuple[int,...] = (6, 8, 8, 8),
#                 num_init_features: int = 64,
#                 bn_size: int = 4,
#                 drop_rate: float = 0.0,
#                 num_classes: int = 14):
#        super().__init__()
#        self.block_config = block_config
#        self.stem = nn.Sequential(
#            nn.Conv2d(in_ch, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
#            nn.BatchNorm2d(num_init_features),
#            nn.ReLU(inplace=False),
##            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#        )

        # Dense blocks
#        features = []
 #       nfeat = num_init_features
#        for i, nl in enumerate(block_config):
#            block = _DenseBlock(nl, nfeat, bn_size, growth_rate, drop_rate)
#            features += [block]
#            nfeat = nfeat + nl*growth_rate
#            if i != len(block_config) - 1:
#                trans = _Transition(nfeat, nfeat // 2)
#                features += [trans]
#                nfeat = nfeat // 2

#        self.features = nn.Sequential(*features, nn.BatchNorm2d(nfeat))
#        self.relu_head = nn.ReLU(inplace=False)
#        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#        self.classifier = nn.Linear(nfeat, num_classes)   # ??? logits(?? Sigmoid)

        # init
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight)
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)
#            elif isinstance(m, nn.Linear):
#                nn.init.constant_(m.bias, 0.0)

#    def forward(self, x):
#        x = self.stem(x)
#        x = self.features(x)
#        x = self.relu_head(x)
#        x = self.avgpool(x)
#        x = torch.flatten(x, 1)
#        logits = self.classifier(x)   # logits;Sigmoid ?? host ??
#        return logits

# --------- ??:?? Sigmoid / ?? ReLU ? in-place ---------
#def prepare_for_dpu(model: nn.Module):
#    import torch.nn as nn
#    for name, m in model.named_modules():
#        if isinstance(m, nn.ReLU):
#            m.inplace = False
    # strip ????? Sigmoid
#    def _strip(mod):
#        for n, ch in list(mod.named_children()):
#            if isinstance(ch, nn.Sigmoid):
##                setattr(mod, n, nn.Identity())
#            else:
#                _strip(ch)
#    _strip(model)
#    return model

# --------- ??:???????(??? head=Linear?Sigmoid)---------
#def load_float_weights_compat(model: nn.Module, ckpt_path: str):
#    state = torch.load(ckpt_path, map_location="cpu")

    # ?? torchvision ??(features.conv0.* ??)? ??????
#    if any(k.startswith("features.conv0.") for k in state.keys()):
#        bc = getattr(model, "block_config", (6,8,8,8))
#        state = _remap_tv_to_custom(state, bc)
#    else:
        # ??????:?? head Sequential ? Linear
#        if "classifier.0.weight" in state:
#            state["classifier.weight"] = state.pop("classifier.0.weight")
#        if "classifier.0.bias" in state:
#            state["classifier.bias"] = state.pop("classifier.0.bias")
#        # conv0 3?1(??? ckpt ? 3ch)
#        if "stem.0.weight" in state and model.stem[0].in_channels == 1:
#            w = state["stem.0.weight"]
#            if w.ndim == 4 and w.shape[1] == 3:
#                state["stem.0.weight"] = w.mean(dim=1, keepdim=True)

 #   miss, unexp = model.load_state_dict(state, strict=False)
#    print("load_state:", "missing", miss, "| unexpected", unexp)

# --------- ????(?? / ???)---------
#def build_model_for_quant(num_classes=14, ckpt=None):
#    model = DenseNet68(in_ch=1, num_classes=num_classes)
#    if ckpt:
#        load_float_weights_compat(model, ckpt)
#    model = prepare_for_dpu(model).eval()
#    return model
