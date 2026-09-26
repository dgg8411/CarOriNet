import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import configs.config as config
from src.models.classifier import OrientationClassifier

# --- 配置参数 ---

# 假设的输入图像大小
INPUT_SIZE = 224

# 使用 CPU 或 GPU
DEVICE = "cpu"
print(f"使用的设备: {DEVICE}")


# --- 1. 模型加载与准备 ---


# --- 2. 数据预处理 ---

def preprocess_image(image):
    """加载并预处理图像，使其适合 ResNet18 输入。"""

    # 训练时使用的标准 ImageNet 归一化参数
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # 假设我们从文件中加载一张 RGB 图像

    # 应用变换
    input_tensor = transform(image)

    # 增加批次维度 (1, C, H, W)，并移动到设备
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    return input_batch


# --- 3. 执行推理 ---

def run_inference(model, input_batch, class_names, top_k=3):
    """执行模型推理并解析结果。"""
    if input_batch is None:
        return

    with torch.no_grad():
        # 前向传播
        output = model(input_batch)

        # 转换为概率
        probabilities = F.softmax(output[0], dim=0)

        # 获取 Top-K 结果
        top_p, top_indices = probabilities.topk(top_k, dim=0)

    # 打印和返回结果
    print("\n--- 推理结果 ---")
    predicted_index = top_indices[0].item()
    predicted_name = class_names[predicted_index]
    predicted_prob = top_p[0].item()

    print(f"预测类别名称: {predicted_name}")
    print(f"预测概率: {predicted_prob:.4f}")

    if top_k > 1:
        print(f"\nTop-{top_k} 预测:")
        for i in range(top_k):
            index = top_indices[i].item()
            prob = top_p[i].item()
            print(f"  {class_names[index]}: {prob:.4f}")


device = "cpu"
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# --- 主执行区 ---
if __name__ == "__main__":

    # 模拟创建一张输入图片
    classes = ['00-背面', '01-右侧面', '02-右后侧', '03-右前侧', '04-正面', '05-左侧面', '06-左后侧', '07-左前侧']

    print(classes)
    model = OrientationClassifier(len(classes), pretrained=False)
    state_dict = torch.load('output/swin_t/orientation_classification_swin_t_50_epochs.pth', map_location='cpu')

    # 4. 将状态字典加载到模型中
    # strict=True: 严格要求 state_dict 中的所有键与 model 中的参数键完全匹配
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    start = time.time()
    model = torch.compile(model, backend='openvino')
    end = time.time()
    print(f"模型编译执行时间: {end - start:.6f} 秒")

    image = Image.open('imgs/屏幕截图 2025-12-02 194702.png').convert('RGB')

    # 2. 准备数据
    input_tensor = preprocess_image(image)
    # input_tensor = input_tensor.to(device)

    for i in range(5):
        start = time.time()

        # 3. 执行推理
        if input_tensor is not None:
            run_inference(model, input_tensor, classes, top_k=5)

        end = time.time()
        print(f"第{i + 1}次执行时间: {end - start:.6f} 秒")