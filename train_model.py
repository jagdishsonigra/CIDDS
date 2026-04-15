import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProcessedDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img = torch.tensor(item["image"], dtype=torch.float32).permute(2, 0, 1) / 255.0

        boxes = torch.tensor(item["boxes"], dtype=torch.float32)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.tensor(item["labels"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target


train_dataset = ProcessedDataset("processed_data/train.pt")
val_dataset = ProcessedDataset("processed_data/valid.pt")
test_dataset = ProcessedDataset("processed_data/test.pt")

collate_fn = lambda x: tuple(zip(*x))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    2  # background + damage
)

model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


def evaluate(loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for i in range(len(outputs)):
                preds = outputs[i]["boxes"].cpu().numpy()
                gts = targets[i]["boxes"].numpy()

                for gt in gts:
                    total += 1
                    matched = False

                    for pred in preds:
                        if compute_iou(pred, gt) > 0.5:
                            matched = True
                            break

                    if matched:
                        correct += 1

    return correct / total if total > 0 else 0


EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_score = evaluate(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Training Loss: {total_loss:.4f}")
    print(f"Validation IoU: {val_score:.4f}")


torch.save(model.state_dict(), "damage_model.pth")

test_score = evaluate(test_loader)

print("\n🚀 FINAL RESULT")
print(f"Test IoU Accuracy: {test_score:.4f}")
print("✅ TRAINING COMPLETE")