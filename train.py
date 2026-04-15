from ultralytics import YOLO
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Using device: {device}")

DATA_YAML = "data.yaml"
PROJECT_NAME = "container_damage"


MODEL = "yolov8n.pt"

model = YOLO(MODEL)
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=4,
    device=device,
    workers=2,
    cache=True,
    amp=False,
    project="runs",
    name=PROJECT_NAME,
    val=True,
    plots=True
)

print("✅ Training Complete!")


best_model = YOLO(f"runs/detect/{PROJECT_NAME}/weights/best.pt")

print("\n📊 Running Test Evaluation...")

test_results = best_model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=4,
    device=device
)

print("\n📊 TEST RESULTS:")
print(test_results)

print("\n🖼️ Generating Predictions...")

best_model.predict(
    source="/Users/jagdishsonigra/PycharmProjects/PBEL/test/images",
    conf=0.4,
    save=True,
    device=device
)

print("\n🎉 ALL DONE!")