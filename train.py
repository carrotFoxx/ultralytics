from ultralytics import YOLO
from roboflow import Roboflow
# rf = Roboflow(api_key="oMEl9snOzhEBNqwD3kHM")
# project = rf.workspace("peter-androids-7aish").project("navi-8pizx")
# dataset = project.version(1).download("yolov8", "datasets/paper-cup")

model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
# model.train(data="coco128.yaml", epochs=10, imgsz=640)
model.train(data="paper-cap.yaml", epochs=10, imgsz=640)

