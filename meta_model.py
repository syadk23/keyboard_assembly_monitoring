import pandas as pd
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import cv2
import os

models = [
    YOLO("runs/detect/train15/weights/best.pt"),
    YOLO('runs/detect/train3/weights/best.pt'),
    YOLO('best.pt')
]

def extract_features(boxes_all, scores_all, labels_all, image_shape):
    rows = []
    num_models = len(models)

    # Flatten all boxes
    all_detections = []
    for model_index, (boxes, scores, labels) in enumerate(zip(boxes_all, scores_all, labels_all)):
        for i in range(len(boxes)):
            b = boxes[i]
            x1 = int(b[0] * image_shape[1])
            y1 = int(b[1] * image_shape[0])
            x2 = int(b[2] * image_shape[1])
            y2 = int(b[3] * image_shape[0])
            conf = scores[i]
            label = labels[i]
            area = (x2 - x1) * (y2 - y1)
            ar = (x2 - x1) / max((y2 - y1), 1)

            all_detections.append((x1, y1, x2, y2, conf, area, ar, label, model_index))

    # Weak labeling via heuristic
    for det in all_detections:
        x1, y1, x2, y2, conf, area, ar, label, model_idx = det

        overlaps = 0
        for other in all_detections:
            if other == det:
                continue
            # IoU-based check
            xa = max(x1, other[0])
            ya = max(y1, other[1])
            xb = min(x2, other[2])
            yb = min(y2, other[3])
            inter_area = max(0, xb - xa) * max(0, yb - ya)
            union_area = area + other[5] - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > 0.5:
                overlaps += 1

        # Heuristic label
        label_class = 1 if overlaps >= 1 and conf > 0.5 else 0

        rows.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'conf': conf, 'area': area, 'aspect_ratio': ar,
            'label': label_class
        })

    return rows

def collect_dataset_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes_all, scores_all, labels_all = [], [], []
        for model in models:
            results = model(frame)[0]
            boxes, scores, labels = [], [], []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.item())
                cls = int(box.cls.item())
                box_norm = [x1 / frame.shape[1], y1 / frame.shape[0],
                            x2 / frame.shape[1], y2 / frame.shape[0]]
                boxes.append(box_norm)
                scores.append(conf)
                labels.append(cls)
            boxes_all.append(boxes)
            scores_all.append(scores)
            labels_all.append(labels)

        features = extract_features(boxes_all, scores_all, labels_all, frame.shape)
        data.extend(features)

    cap.release()
    return pd.DataFrame(data)


from sklearn.ensemble import RandomForestClassifier
import joblib

df = collect_dataset_from_video("test.mp4")

X = df[['x1', 'y1', 'x2', 'y2', 'conf', 'area', 'aspect_ratio']]
y = df['label']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save meta-model
joblib.dump(model, "meta_model.pkl")
print("✅ Saved meta_model.pkl")
