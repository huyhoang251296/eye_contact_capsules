import pandas as pd
from tqdm import tqdm
from PIL import Image
import os


def get_data(data_dir="./dataset"):
    image_extensions = []

    image_files = os.listdir(os.path.join(data_dir, "images"))
    image_extensions = list(map(lambda x: x.split(".")[-1], image_files))

    eye_contact_ds = {
        "paths": [],
        "labels": []
    }

    outliers = []

    label_dir = os.path.join(data_dir, "labels")
    label_files = os.listdir(label_dir)

    for fn in tqdm(label_files):
        file_path = os.path.join(label_dir, fn)
        image_fn = fn.split(".")[0] + ".jpg"
        if image_fn not in image_files:
            outliers.append(file_path)
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        img_path = os.path.join(data_dir, "images", image_fn)
        eye_contact_ds["paths"].append(img_path)
        eye_contact_ds["labels"].append(int(data[:2] == "10"))

    return eye_contact_ds