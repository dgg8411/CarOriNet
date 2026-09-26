import torch
import torch.nn as nn
from torchvision import models

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import configs.config as config


class OrientationClassifier(nn.Module):
    """
    一个通用的分类模型封装器，用于加载预训练模型并自动替换其分类头。

    该类尝试识别并替换模型中常见的分类层 (如 ViT 的 heads.head, ResNet 的 fc, 或其它模型的 classifier)。
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        初始化模型并修改分类器。

        Args:
            num_classes (int): 目标数据集的类别数量。
            pretrained (bool): 是否加载预训练权重。
        """
        super(OrientationClassifier, self).__init__()
        model = models.swin_t(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_classes)

        # 冻结所有主干参数（只保留分类头可训练）
        for param in model.parameters():
            param.requires_grad = True

        # 解冻分类头（确保它可训练）
        for param in model.head.parameters():
            param.requires_grad = True

        # 1. 加载预训练的基础模型
        self.base_model = model

    def forward(self, x):
        """
        执行模型的前向传播。
        """
        return self.base_model(x)


def load_model(num_classes, device, pretrained_model_path=None):
    model = OrientationClassifier(num_classes, pretrained=True)
    pretrained_model_path = pretrained_model_path or getattr(config, 'PRETRAINED_MODEL', None)
    if pretrained_model_path:
        state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # 4. 将状态字典加载到模型中
        # strict=True: 严格要求 state_dict 中的所有键与 model 中的参数键完全匹配
        model.load_state_dict(state_dict, strict=True)
    return model.to(device)


if __name__ == "__main__":
    model = OrientationClassifier(8)