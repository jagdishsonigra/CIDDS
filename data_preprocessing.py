import os
import torch
import cv2

DATASETS = {
    "train": ("train/images", "train/labels"),
    "valid": ("valid/images", "valid/labels"),
    "test": ("test/images", "test/labels"),
}

OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_split(split_name, img_dir, label_dir):
    data = []

    for file in os.listdir(img_dir):
        if not file.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(label_dir, file.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    cls, x, y, bw, bh = map(float, line.split())

                    xmin = (x - bw/2) * w
                    xmax = (x + bw/2) * w
                    ymin = (y - bh/2) * h
                    ymax = (y + bh/2) * h

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)

        data.append({
            "image": img,
            "boxes": boxes,
            "labels": labels
        })

    save_path = os.path.join(OUTPUT_DIR, f"{split_name}.pt")
    torch.save(data, save_path)

    print(f"✅ {split_name} saved → {save_path}")

for split, (img_dir, label_dir) in DATASETS.items():
    process_split(split, img_dir, label_dir)

print("\n🎯 DATA PREPARATION COMPLETE")