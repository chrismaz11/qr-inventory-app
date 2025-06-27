from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # download + load YOLOv8n model
model.export(format='onnx', opset=12, imgsz=[640, 640])  # export to yolov8n.onnx
