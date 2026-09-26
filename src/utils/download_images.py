import io
import os
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from loguru import logger


def download_img(url, retry_times=3):
    for _ in range(retry_times):
        try:
            logger.info(f"开始执行 download_img {url}")
            response = requests.get(url, stream=True)
            with io.BytesIO(response.content) as b:
                image = Image.open(b).copy()

            logger.info(f"已完成 download_img {url}")
            return image
        except Exception as e:
            logger.error(e)


def find_file_in_subdirs(root_dir: str, target_filename: str) -> bool:
    """
    在 root_dir 及其所有子目录中查找是否存在 target_filename。
    找到立即返回 True，未找到返回 False。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            return True
    return False


def sale_to_res(org_size, res=512):
    w = org_size[0]
    h = org_size[1]

    scale = w / h

    rx = (res ** 2 / scale) ** 0.5

    return int(round(scale * rx, 0)), int(round(rx, 0))


if __name__ == "__main__":
    df = pd.read_csv('全车系图.csv').head(20000)
    output_path = r"I:\方向数据集\orientation-dataset-20k"