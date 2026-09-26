import argparse
import importlib
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataloader import get_dataloaders
from src.models.classifier import load_model

# 载入配置
config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train_information():
    parser = argparse.ArgumentParser(description='Iron Defect Detection')
    parser.add_argument('--model', default=config.MODEL_NAME, type=str)
    parser.add_argument('--device', default=device, type=str)
    parser.add_argument('--epoch', default=config.EPOCHS, type=int, help="training epochs")
    parser.add_argument('--dataset_path', default=config.DATASET_PATH, type=str)
    parser.add_argument('--batchsize', default=config.BATCH_SIZE, type=int)
    parser.add_argument('--init_lr', default=config.INIT_LR, type=float)
    args = parser.parse_args(args=[])
    print(args)

def compute_class_weights(dataset):
    labels = [sample[1] for sample in dataset]
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total = sum(class_counts.values())
    weights = torch.tensor(
        [total / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float
    ).to(device)
    return weights


def evaluate(model, dataloader, criterion, epoch=None, desc="Val"):
    model.eval()
    total, correct = 0, 0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            with autocast(device_type="cuda"):
                out = model(x)
                loss = criterion(out, y)
            _, predicted = torch.max(out, 1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            preds.extend(predicted.cpu().numpy())
            targets.extend(y.cpu().numpy())

    acc = 100 * correct / total
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)

    if epoch is not None:
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        print(f"[{desc}][Epoch {epoch+1:03d}] Acc: {acc:.2f}% | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
    else:
        print(f"[{desc}][Final] Acc: {acc:.2f}% | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
        print(f"Confusion Matrix:\n{confusion_matrix(targets, preds)}")

    return acc

val_precisions = []
val_recalls = []
val_f1s = []

def train():
    train_information()
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    model = load_model(len(classes),device)

    print(f"训练标签类别: {classes}")
    class_weights = compute_class_weights(train_loader.dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=config.INIT_LR, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler()

    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                out = model(x)
                loss = criterion(out, y)
            _, predicted = torch.max(out, 1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_acc = 100 * correct / total
        print(f"[Train][Epoch {epoch+1:03d}] Acc: {train_acc:.2f}%")

        val_acc = evaluate(model, val_loader, criterion, epoch, desc="Val")
        if val_acc > best_acc:
            best_acc = val_acc
            if not os.path.exists(config.OUTPUT_DIR):
                os.makedirs(config.OUTPUT_DIR)
            torch.save(model.state_dict(), config.OUTPUT_MODEL)
            print(f"最佳模型已保存（准确率: {val_acc:.2f}%）")

    print("开始在测试集上评估最终模型...")
    model.load_state_dict(torch.load(config.OUTPUT_MODEL))
    evaluate(model, test_loader, criterion, desc="Test")

    # 保存precision/recall/F1图像
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    epochs = range(1, config.EPOCHS + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_precisions, label='Precision', marker='o')
    plt.plot(epochs, val_recalls, label='Recall', marker='s')
    plt.plot(epochs, val_f1s, label='F1-score', marker='^')
    plt.title('Validation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, f"metrics_{config.EPOCHS}epochs.png"))
    print("指标可视化已保存")


if __name__ == "__main__":
    train()