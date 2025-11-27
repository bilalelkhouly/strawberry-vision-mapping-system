import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ["ULTRALYTICS_AGREE"] = "true"
from ultralytics import YOLO

DATA_YAML = "/content/dataset.yaml"
MODEL     = "yolov8n.pt"
PROJECT   = "run_strawberry_labelled_detection"
RUN_NAME  = "strawberry_flower_detection_model"

model = YOLO(MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    workers=2,
    amp=True,
    project=PROJECT,
    name=RUN_NAME,
    patience=20,
    cache=False,
    deterministic=True,
)