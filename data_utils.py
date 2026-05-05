import os
import cv2 as cv
import numpy as np
from tqdm import tqdm


def count_images(base_dir: str, split: str):
    valid_extensions = (".jpeg", ".jpg", ".png")
    counts = {}

    for class_name in ["NORMAL", "PNEUMONIA"]:
        class_dir = os.path.join(base_dir, split, class_name)
        files = os.listdir(class_dir)
        image_files = [file for file in files if file.lower().endswith(valid_extensions)]
        counts[class_name] = len(image_files)

    return counts


def collect_data(base_dir: str, split: str = "train"):
    normal_label = 1
    pneumonia_label = -1

    images = []
    labels = []

    normal_path = os.path.join(base_dir, split, "NORMAL")

    for img_file in tqdm(os.listdir(normal_path), desc=f"Loading {split} - NORMAL"):
        img_path = os.path.join(normal_path, img_file)
        img = cv.imread(img_path)

        if img is not None:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
            img = img.reshape(-1)
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(normal_label)

    pneumonia_path = os.path.join(base_dir, split, "PNEUMONIA")

    for img_file in tqdm(os.listdir(pneumonia_path), desc=f"Loading {split} - PNEUMONIA"):
        img_path = os.path.join(pneumonia_path, img_file)
        img = cv.imread(img_path)

        if img is not None:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
            img = img.reshape(-1)
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(pneumonia_label)

    return np.stack(images, axis=0), np.array(labels)