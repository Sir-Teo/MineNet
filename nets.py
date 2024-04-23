import torch
import torch.nn as nn
from torchvision.models import resnet50, densenet121, densenet169, densenet201, efficientnet_b0, efficientnet_b7, vit_b_16, swin_t, convnext_tiny
import sys
import os
sys.path.append('VMamba/classification')
import yaml

from models.vmamba import VSSM
from helpers import NestedNamespace


class VSSM_Model(nn.Module):
    def __init__(self, num_classes, num_channels,loss="bce", use_weights=True):
        super(VSSM_Model, self).__init__()
        if use_weights:
            checkpoint = torch.load('/scratch/wz1492/MineNet/VMamba/weights/vssm_base_0229_ckpt_epoch_237.pth')
            with open("/scratch/wz1492/MineNet/VMamba/weights/vssm_base_224.yaml") as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            config = NestedNamespace(config)
        self.model = VSSM(
            num_classes=num_classes,
            num_channels=num_channels,
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        )

        
        self.model.load_state_dict(checkpoint['model'], strict=False)

        for p in self.model.parameters():
            p.requires_grad = True
        if loss == "crossentropy":
            self.model.classifier.head = nn.Linear(self.model.num_features, num_classes)
        elif loss == "bce":
            self.model.classifier.head = nn.Sequential(
                nn.Linear(self.model.num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        

    def forward(self, x):
        x = self.model(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True):
        super(VisionTransformer, self).__init__()
        self.model = vit_b_16(weights=("DEFAULT" if use_weights else None), image_size=224)
        self.model.conv_proj = nn.Conv2d(num_channels, 768, kernel_size=16, stride=16)
        
        if loss == "crossentropy":
            self.model.heads = nn.Linear(768, num_classes)
        elif loss == "bce":
            self.model.heads = nn.Linear(768, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True):
        super(SwinTransformer, self).__init__()
        self.model = swin_t(weights=("DEFAULT" if use_weights else None))
        self.model.features[0][0] = nn.Conv2d(num_channels, 96, kernel_size=4, stride=4)
        
        if loss == "crossentropy":
            self.model.head = nn.Linear(768, num_classes)
        elif loss == "bce":
            self.model.head = nn.Linear(768, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True):
        super(ConvNeXt, self).__init__()
        self.model = convnext_tiny(weights=("DEFAULT" if use_weights else None))
        self.model.features[0][0] = nn.Conv2d(num_channels, 96, kernel_size=4, stride=4)
        
        if loss == "crossentropy":
            self.model.classifier[2] = nn.Linear(768, num_classes)
        elif loss == "bce":
            self.model.classifier[2] = nn.Linear(768, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True, weight_decay=1e-4, dropout_prob=0.5):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=("DEFAULT" if use_weights else None))
        self.model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_ftrs = self.model.fc.in_features
        if loss == "crossentropy":
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif loss == "bce":
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        x = self.model(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True):
        super(DenseNet121, self).__init__()
        self.model = densenet121(weights=("DEFAULT" if use_weights else None))
        self.model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.classifier.in_features
        
        if loss == "crossentropy":
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
        elif loss == "bce":
            self.model.classifier = nn.Linear(num_ftrs, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x

class DenseNet169(nn.Module):
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True):
        super(DenseNet169, self).__init__()
        self.model = densenet169(weights=("DEFAULT" if use_weights else None))
        self.model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.classifier.in_features
        
        if loss == "crossentropy":
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
        elif loss == "bce":
            self.model.classifier = nn.Linear(num_ftrs, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x

class DenseNet201(nn.Module):
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True):
        super(DenseNet201, self).__init__()
        self.model = densenet201(weights=("DEFAULT" if use_weights else None))
        self.model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.classifier.in_features
        
        if loss == "crossentropy":
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
        elif loss == "bce":
            self.model.classifier = nn.Linear(num_ftrs, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True):
        super(EfficientNetB0, self).__init__()
        self.model = efficientnet_b0(weights=("DEFAULT" if use_weights else None))
        self.model.features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        num_ftrs = self.model.classifier[1].in_features
        
        if loss == "crossentropy":
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif loss == "bce":
            self.model.classifier[1] = nn.Linear(num_ftrs, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x

class EfficientNetB7(nn.Module):
    # out of memory error
    def __init__(self, num_classes, num_channels, loss="bce", use_weights=True):
        super(EfficientNetB7, self).__init__()
        self.model = efficientnet_b7(weights=("DEFAULT" if use_weights else None))
        self.model.features[0][0] = nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        num_ftrs = self.model.classifier[1].in_features
        
        if loss == "crossentropy":
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif loss == "bce":
            self.model.classifier[1] = nn.Linear(num_ftrs, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x