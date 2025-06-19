import cv2
import joblib
import numpy as np
from task_tracking import *
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

val_dir = 'dataset/valid'

models = [
    YOLO("runs/detect/train15/weights/best.pt"),
    YOLO('runs/detect/train3/weights/best.pt'),
    YOLO('best.pt')
]

model_class_names = [
  {0: "0"},
  {1: "1"},
  {2: "2"},
  {3: "3"},
  {4: "4"},
  {5: "5"},
  {6: "6"},
  {7: "7"},
  {8: "8"},
  {9: "9"},
  {10: "a"},
  {11: "accent"},
  {12: "ae"},
  {13: "alt-left"},
  {14: "altgr-right"},
  {15: "b"},
  {16: "c"},
  {17: "caret"},
  {18: "comma"},
  {19: "d"},
  {20: "del"},
  {21: "e"},
  {22: "enter"},
  {23: "f"},
  {24: "g"},
  {25: "h"},
  {26: "hash"},
  {27: "i"},
  {28: "j"},
  {29: "k"},
  {30: "keyboard"},
  {31: "l"},
  {32: "less"},
  {33: "m"},
  {34: "minus"},
  {35: "n"},
  {36: "o"},
  {37: "oe"},
  {38: "p"},
  {39: "plus"},
  {40: "point"},
  {41: "q"},
  {42: "r"},
  {43: "s"},
  {44: "shift-left"},
  {45: "shift-lock"},
  {46: "shift-right"},
  {47: "space"},
  {48: "ss"},
  {49: "strg-left"},
  {50: "strg-right"},
  {51: "t"},
  {52: "tab"},
  {53: "u"},
  {54: "ue"},
  {55: "v"},
  {56: "w"},
  {57: "x"},
  {58: "y"},
  {59: "z"},
  {60: "face"},
  {61: "person"},
  {62: "key-switch"},
  {63: "hand"},
  {64: "case"},
  {65: "pcb"}
]

stacking_model = joblib.load("meta_model.pkl")

def run_models(frame):
    model_class_names = []
    for model in models:
        if hasattr(model.model, 'names'):
            model_class_names.append(model.model.names)
        else:
            model_class_names.append({})  # fallback if not available

    boxes_all, scores_all, labels_all = [], [], []
    for i, model in enumerate(models):
        results = model(frame)[0]
        boxes, scores, labels = [], [], []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.item())
            class_id = int(box.cls.item())
            label_name = model_class_names[i].get(class_id, f"unknown_cls{class_id}")

            # Normalize coordinates for WBF
            box_norm = [x1 / frame.shape[1], y1 / frame.shape[0],
                        x2 / frame.shape[1], y2 / frame.shape[0]]

            boxes.append(box_norm)
            scores.append(conf)
            labels.append(label_name)

        boxes_all.append(boxes)
        scores_all.append(scores)
        labels_all.append(labels)

    return boxes_all, scores_all, labels_all

def fuse_and_stack(boxes_all, scores_all, labels_all, image_shape):
    # Convert class name labels to integers for WBF
    all_labels = list(set(l for labels in labels_all for l in labels))
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    numeric_labels_all = [
        [label_to_id[label] for label in labels]
        for labels in labels_all
    ]

    # Apply WBF
    boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
        boxes_all, scores_all, numeric_labels_all,
        iou_thr=0.4, skip_box_thr=0.1
    )

    final_detections = []
    for i in range(len(boxes_fused)):
        x1 = int(boxes_fused[i][0] * image_shape[1])
        y1 = int(boxes_fused[i][1] * image_shape[0])
        x2 = int(boxes_fused[i][2] * image_shape[1])
        y2 = int(boxes_fused[i][3] * image_shape[0])
        score = scores_fused[i]
        label_id = labels_fused[i]
        class_name = id_to_label[label_id]

        # Meta-model feature: [x1, y1, x2, y2, confidence, area, aspect_ratio]
        area = (x2 - x1) * (y2 - y1)
        ar = (x2 - x1) / max((y2 - y1), 1)
        feature_vec = np.array([[x1, y1, x2, y2, score, area, ar]])

        # Predict class using meta-model
        pred_label = stacking_model.predict(feature_vec)[0]
        if hasattr(stacking_model, 'predict_proba'):
            proba = stacking_model.predict_proba(feature_vec)[0]
            confidence = max(proba)
        else:
            confidence = score  # fallback to original WBF score

        final_detections.append((x1, y1, x2, y2, confidence, class_name))

    return final_detections

def main():
    # Train
    #model.train(data="dataset/data.yaml", epochs=20, imgsz=640)

    assembly_tasks = Task("Keyboard Assembly", [
        "Mount PCB",
        "Install stabilizers",
        "Insert switches",
        "Install keycaps"
    ])

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes_all, scores_all, labels_all = run_models(frame)
        detections = fuse_and_stack(boxes_all, scores_all, labels_all, frame.shape)

        for (x1, y1, x2, y2, conf, label) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
       
        cv2.imshow("Keyboard Assembly Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()