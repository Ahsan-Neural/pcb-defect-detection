import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Class definitions
# ─────────────────────────────────────────────
CLASS_LABELS = [
    "open_circuit",
    "short_circuit",
    "mouse_bite",
    "spur",
    "copper_spill",
    "pin_hole"
]

CLASS_COLORS = {
    "open_circuit"  : (231, 76,  60),
    "short_circuit" : (52,  152, 219),
    "mouse_bite"    : (46,  204, 113),
    "spur"          : (243, 156, 18),
    "copper_spill"  : (155, 89,  182),
    "pin_hole"      : (26,  188, 156),
}

# ─────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────
def predict(
    image_path: str,
    model_path: str = "models/best.pt",
    conf: float = 0.25,
    iou: float  = 0.45,
    device: str = "cpu"
) -> dict:
    """
    Run YOLOv8 PCB defect detection on a single image.

    Args:
        image_path : Path to input image (.jpg / .png)
        model_path : Path to YOLOv8 best.pt weights
        conf       : Confidence threshold (0.0 – 1.0)
        iou        : IoU threshold for NMS (0.0 – 1.0)
        device     : 'cpu' or '0' for GPU

    Returns:
        dict:
            'detections' → list of {class, confidence, bbox}
            'count'      → total defects detected
            'annotated'  → annotated image as numpy RGB array
            'per_class'  → defect count per class
    """
    model   = YOLO(model_path)
    results = model.predict(
        image_path,
        conf    = conf,
        iou     = iou,
        device  = device,
        verbose = False
    )[0]

    # ── Parse detections ──────────────────────
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_name = CLASS_LABELS[int(box.cls[0])]
            detections.append({
                "class"      : cls_name,
                "confidence" : round(float(box.conf[0]), 4),
                "bbox"       : [round(v, 2) for v in
                                box.xyxy[0].cpu().tolist()]
            })

    # ── Count per class ───────────────────────
    per_class = {cls: 0 for cls in CLASS_LABELS}
    for det in detections:
        per_class[det["class"]] += 1

    # ── Draw annotations ──────────────────────
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = CLASS_COLORS[det["class"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        cv2.rectangle(img,
                      (x1, max(0, y1 - th - 6)),
                      (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label,
                    (x1 + 2, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)

    return {
        "detections" : detections,
        "count"      : len(detections),
        "annotated"  : img,
        "per_class"  : per_class
    }
