import os
import cv2
import numpy as np
import torch
import time
import random
from contextlib import redirect_stdout, redirect_stderr
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

VAL_DIR = "dataset/valid_together"  # folder with images and .txt labels

# By default, don't print per-inference logs from the models.
VERBOSE = False

# Load models
models = [
    YOLO("human_model.pt"),
    YOLO('screw_model.pt'),
    YOLO('keyboard_model.pt')
]

model_class_names = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "a", 11: "accent", 12: "ae", 13: "alt-left", 14: "altgr-right", 15: "b",
    16: "c", 17: "caret", 18: "comma", 19: "d", 20: "del", 21: "e", 22: "enter", 23: "f",
    24: "g", 25: "h", 26: "hash", 27: "i", 28: "j", 29: "k", 30: "keyboard", 31: "l",
    32: "less", 33: "m", 34: "minus", 35: "n", 36: "o", 37: "oe", 38: "p", 39: "plus",
    40: "point", 41: "q", 42: "r", 43: "s", 44: "shift-left", 45: "shift-lock",
    46: "shift-right", 47: "space", 48: "ss", 49: "strg-left", 50: "strg-right",
    51: "t", 52: "tab", 53: "u", 54: "ue", 55: "v", 56: "w", 57: "x", 58: "y", 59: "z",
    60: "face", 61: "person", 62: "key-switch", 63: "hand", 64: "case", 65: "pcb", 66: "screw"
}

# Reverse map
id_to_label = {i: name for i, name in model_class_names.items()}

def load_ground_truth(txt_path, img_shape):
    img_height, img_width = img_shape[:2]
    boxes = []
    labels = []

    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) == 9:
                cls_id = int(parts[0])
                coords = parts[1:]

                # Group into (x, y) pairs and scale to pixel coords
                points = [(coords[i] * img_width, coords[i + 1] * img_height) for i in range(0, 8, 2)]

                # Get axis-aligned bounding box
                xs, ys = zip(*points)
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)

                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

    return boxes, labels

def evaluate_model(model, model_name):
    metric = MeanAveragePrecision(iou_type="bbox")
    image_files = [f for f in os.listdir(VAL_DIR) if f.endswith(('.jpg', '.png'))]

    for img_name in image_files:
        image_path = os.path.join(VAL_DIR, img_name)
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        img = cv2.imread(image_path)

        # Inference (suppress model's own prints when VERBOSE is False)
        # Call model with verbose=False to avoid ultralytics printing timings
        if not VERBOSE:
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    results = model(img, verbose=False)[0]
        else:
            results = model(img, verbose=False)[0]
        pred_boxes, pred_scores, pred_labels = [], [], []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.item())
            class_id = int(box.cls.item())
            pred_boxes.append([x1, y1, x2, y2])
            pred_scores.append(conf)
            pred_labels.append(class_id)

        gt_boxes, gt_labels = load_ground_truth(txt_path, img.shape)

        preds = [{
            "boxes": torch.tensor(pred_boxes),
            "scores": torch.tensor(pred_scores),
            "labels": torch.tensor(pred_labels).long()
        }]

        target = [{
            "boxes": torch.tensor(gt_boxes),
            "labels": torch.tensor(gt_labels).long()
        }]

        metric.update(preds, target)

        results = metric.compute()
        precision = results["map_50"]
        recall = results["mar_100"]
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        if VERBOSE:
            print(f"\n📊 Results for {model_name}")
            print(f"Precision (mAP@0.5): {precision:.4f}")
            print(f"Recall (mAR@100):    {recall:.4f}")
            print(f"F1 Score:            {f1:.4f}")

        return precision.item(), recall.item(), f1.item()

def run_all_models(frame):
    boxes_all, scores_all, labels_all = [], [], []
    for i, model in enumerate(models):
        # Call model with verbose=False and additionally suppress stdout/stderr
        if not VERBOSE:
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    results = model(frame, verbose=False)[0]
        else:
            results = model(frame, verbose=False)[0]
        boxes, scores, labels = [], [], []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.item())
            class_id = int(box.cls.item())

            if class_id not in model_class_names:
                continue

            box_norm = [x1 / frame.shape[1], y1 / frame.shape[0],
                        x2 / frame.shape[1], y2 / frame.shape[0]]

            boxes.append(box_norm)
            scores.append(conf)
            labels.append(class_id)

        boxes_all.append(boxes)
        scores_all.append(scores)
        labels_all.append(labels)

    return boxes_all, scores_all, labels_all

def fuse_predictions(boxes_all, scores_all, labels_all, shape):
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_all, scores_all, labels_all, iou_thr=0.5, skip_box_thr=0.3)

    boxes_pixel = []
    for b in boxes:
        x1 = b[0] * shape[1]
        y1 = b[1] * shape[0]
        x2 = b[2] * shape[1]
        y2 = b[3] * shape[0]
        boxes_pixel.append([x1, y1, x2, y2])

    scores = [min(1.0, s * len(boxes_all)) for s in scores]

    return boxes_pixel, scores, labels

def evaluate_ensemble():
    total_tp = total_fp = total_fn = 0
    metric = MeanAveragePrecision(iou_type="bbox")
    image_files = [f for f in os.listdir(VAL_DIR) if f.endswith(('.jpg', '.png'))]

    for img_name in image_files:
        image_path = os.path.join(VAL_DIR, img_name)
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        img = cv2.imread(image_path)

        # Ensemble prediction
        boxes_all, scores_all, labels_all = run_all_models(img)
        pred_boxes, pred_scores, pred_labels = fuse_predictions(boxes_all, scores_all, labels_all, img.shape)

        # Ground truth
        gt_boxes, gt_labels = load_ground_truth(txt_path, img.shape)

        # metric calcs
        """  tp, fp, fn = compute_detection_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels)
        total_tp += tp
        total_fp += fp
        total_fn += fn """

        preds = [{
            "boxes": torch.tensor(pred_boxes),
            "scores": torch.tensor(pred_scores),
            "labels": torch.tensor(pred_labels).long()
        }]

        target = [{
            "boxes": torch.tensor(gt_boxes),
            "labels": torch.tensor(gt_labels).long()
        }]

        """ # DEBUG
        print(f"\nImage: {img_name}")
        print("GT boxes:", gt_boxes)
        print("GT labels:", gt_labels)
        print("Predicted boxes:", [[float(x) for x in box] for box in pred_boxes])
        print("Predicted labels:", pred_labels)

        debug_img = img.copy()
        for (x1, y1, x2, y2), label in zip(pred_boxes, pred_labels):
            cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(debug_img, id_to_label.get(label, str(label)), (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for (x1, y1, x2, y2), label in zip(gt_boxes, gt_labels):
            cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(debug_img, f"GT:{label}", (int(x1), int(y2) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Debug Ensemble Eval", debug_img)
        cv2.waitKey(0) """

        metric.update(preds, target)

    results = metric.compute()
    if VERBOSE:
        print("\n📊 Ensemble Model Evaluation:")
        for k, v in results.items():
            print(f"{k}: {v}")

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        print(f"\n🔍 Precision: {precision:.4f}")
        print(f"🔍 Recall:    {recall:.4f}")
        print(f"🔍 F1 Score:  {f1:.4f}")
    else:
        # Still compute scores even when verbose is off
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

def compute_detection_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    if not pred_boxes or not gt_boxes:
        return 0, len(pred_boxes), len(gt_boxes)  # All preds are FP, all GT are FN

    ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes))

    for i, pred_label in enumerate(pred_labels):
        best_iou, best_j = 0, -1
        for j, gt_label in enumerate(gt_labels):
            if j in matched_gt or pred_label != gt_label:
                continue
            iou = ious[i, j].item()
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn