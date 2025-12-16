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
from metrics import run_all_models, fuse_predictions

# Paths
VAL_DIR = "dataset/valid_together"  # folder with images and .txt labels

assembly_steps = [
    "Step 1: Place keyboard case",
    "Step 2: Install PCB and mounting plate (if applicable)",
    "Step 3: Attach stabilizers",
    "Step 4: Install switches",
    "Step 5: Install keycaps"
]

instruction_images = [
    cv2.imread("instruction_images/case.jpg"),
    cv2.imread("instruction_images/pcb_installed.png"),
    cv2.imread("instruction_images/stabs_installed.webp"),
    cv2.imread("instruction_images/switches_installed.jpg"),
    cv2.imread("instruction_images/final.webp")
]

# Toggle terminal output (set to False to silence detection/evaluation prints)
VERBOSE = True

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

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

def draw_progress_bar(img, progress):
    bar_x, bar_y = 40, 140
    bar_w, bar_h = 400, 35

    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
    filled = int(bar_w * progress)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h), (0,255,0), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 2)

def make_dashboard(step, progress):
    dashboard = np.zeros((550, 700, 3), dtype=np.uint8)

    cv2.putText(dashboard,
                f"Current Task:",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,255,255), 2)

    cv2.putText(dashboard,
                assembly_steps[step],
                (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0), 2)

    draw_progress_bar(dashboard, progress)

    cv2.putText(dashboard,
                f"Progress: {int(progress*100)}%",
                (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2)

    cv2.putText(dashboard,
                "Press 1-5 to simulate detection",
                (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (180,180,180), 1)
    
    img = instruction_images[step]
    if img is not None:
        dashboard[280:280 + img.shape[0], 60:60 + img.shape[1]] = img

    return dashboard

def main():
    TOTAL_STEPS = len(assembly_steps)
    current_step = 0

    random.seed(42) 
    colors = {}
    for i in range(67):
        colors[i] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    for i in range(len(instruction_images)):
        if instruction_images[i] is not None:
            instruction_images[i] = cv2.resize(
                instruction_images[i], (340, 150)
            )

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes_all, scores_all, labels_all = run_all_models(frame)
        pred_boxes, pred_scores, pred_labels = fuse_predictions(
            boxes_all, scores_all, labels_all, frame.shape
        )

        annotated_frame = frame.copy()
        for (x1, y1, x2, y2), label, score in zip(pred_boxes, pred_labels, pred_scores):
            color = colors[label]
            label_text = f"{id_to_label.get(label, str(label))} {score:.2f}"

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                annotated_frame,
                label_text,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            cv2.putText(frame,
                "Simulate detection: Press keys 1-5   |   Q = Quit",
                (40, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)
            
        progress = current_step / (TOTAL_STEPS - 1)
        dashboard = make_dashboard(current_step, progress)
        cv2.imshow("Webcam", annotated_frame)
        cv2.imshow("Assembly Dashboard", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('1'): 
            current_step = 0
        if cv2.waitKey(1) & 0xFF == ord('2'):
            current_step = 1
        if cv2.waitKey(1) & 0xFF == ord('3'):
            current_step = 2
        if cv2.waitKey(1) & 0xFF == ord('4'):
            current_step = 3
        if cv2.waitKey(1) & 0xFF == ord('5'):
            current_step = 4
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()