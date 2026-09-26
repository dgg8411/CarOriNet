import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def split_dataset(root_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """将数据集按指定比例划分为训练集、验证集和测试集。"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例加起来必须为1"

    classes = os.listdir(root_dir)
    for cls in tqdm(classes, desc="分配中"):
        img_paths = [os.path.join(root_dir, cls, img) for img in os.listdir(os.path.join(root_dir, cls))]
        random.shuffle(img_paths)

        n_total = len(img_paths)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        splits = {
            'train': img_paths[:n_train],
            'val': img_paths[n_train:n_train + n_val],
            'test': img_paths[n_train + n_val:]
        }

        for split, paths in splits.items():
            save_dir = os.path.join(output_dir, split, cls)
            os.makedirs(save_dir, exist_ok=True)
            for p in paths:
                shutil.copy(p, save_dir)


def validate_images(root_dir, extension='png'):
    """
    递归查找指定扩展名的文件，并尝试打开验证图像是否损坏。
    返回所有找到的有效文件路径列表。
    """
    files_list = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(f'.{extension}'):
                full_path = os.path.join(dirpath, filename)
                try:
                    Image.open(full_path).convert('RGB')
                    files_list.append(full_path)
                except Exception as e:
                    print(f"无法打开图像: {full_path}, 错误: {e}")

    return files_list


def find_duplicate_filenames(root_dir: str):
    """
    遍历 root_dir 及其子目录，找出所有文件名重复的文件。

    返回: dict {filename: [Path1, Path2, ...]}
    """
    file_map = defaultdict(list)

    for file_path in Path(root_dir).rglob('*'):
        if file_path.is_file():
            file_map[file_path.name].append(file_path)

    duplicates = {name: paths for name, paths in file_map.items() if len(paths) > 1}
    return duplicates


# 用法示例
if __name__ == "__main__":
    # 示例1: 数据集划分
    # split_dataset(
    #     root_dir=r'I:\方向数据集\orientation-dataset-20k-wr',
    #     output_dir=r'I:\方向数据集\orientation-dataset-20k-wr-train',
    #     train_ratio=0.8,
    #     val_ratio=0.1,
    #     test_ratio=0.1
    # )

    # 示例2: 验证图像
    # root_directory = r'I:\方向数据集\orientation-dataset-split'
    # txt_files = validate_images(root_directory, 'png')

    # 示例3: 查找重复文件名
    root = r"I:\方向数据集\orientation-dataset-20k"
    duplicates = find_duplicate_filenames(root)

    if duplicates:
        print(f"发现 {len(duplicates)} 个重复文件名：\n")
        for filename, paths in duplicates.items():
            print(f"文件名: {filename}")
            for p in paths:
                print(f"   -> {p}")
            print("-" * 50)
    else:
        print("未发现重复文件名。")