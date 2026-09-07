from ultralytics import YOLO
import cv2
import math
import sys
import numpy as np
import json
from datetime import datetime
from pathlib import Path


# ============================================================
# MINEGUARD V3
# VISIBILITY-AWARE RISK ENGINE + EVENT LOGGING
# ============================================================

MODEL_PATH = "runs/detect/training/mining_v1/weights/best.pt"

CONF_THRESHOLD = 0.25

VEHICLE_CLASSES = {
    "dump_truck",
    "hd_truck",
    "mining_truck",
    "excavator",
}

# Directory where MineGuard safety events are stored
EVENT_DIR = Path("mineguard_events")
EVENT_DIR.mkdir(exist_ok=True)


# ============================================================
# INPUT
# ============================================================

if len(sys.argv) < 2:
    print()
    print("Usage:")
    print("python risk_engine.py <image_path>")
    print()
    sys.exit(1)

IMAGE_PATH = sys.argv[1]


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

image_h, image_w = image.shape[:2]


# ============================================================
# VISIBILITY ESTIMATION
# ============================================================
# Prototype image-quality estimation.
#
# IMPORTANT:
# This does NOT estimate visibility in metres.
#
# Production MineGuard would use validated camera analytics
# and complementary environmental/ranging sensors.
# ============================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Image contrast
contrast = float(np.std(gray))

# Image sharpness
laplacian_variance = float(
    cv2.Laplacian(gray, cv2.CV_64F).var()
)


def estimate_visibility(contrast, sharpness):

    if contrast < 30 and sharpness < 100:
        return "LOW", 3

    elif contrast < 45 or sharpness < 250:
        return "REDUCED", 2

    else:
        return "NORMAL", 1


visibility_state, visibility_level = estimate_visibility(
    contrast,
    laplacian_variance
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# YOLO DETECTION
# ============================================================

results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=CONF_THRESHOLD,
    verbose=False
)

result = results[0]


# ============================================================
# EXTRACT VEHICLES
# ============================================================

detections = []

for box in result.boxes:

    cls_id = int(box.cls[0])
    confidence = float(box.conf[0])

    class_name = model.names[cls_id]

    # Ignore non-vehicle classes
    if class_name not in VEHICLE_CLASSES:
        continue

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    width = x2 - x1
    height = y2 - y1

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    area_ratio = (
        (width * height)
        / (image_w * image_h)
    )

    detections.append({
        "class": class_name,
        "confidence": confidence,
        "bbox": (x1, y1, x2, y2),
        "center": (center_x, center_y),
        "area_ratio": area_ratio
    })


# ============================================================
# IOU CALCULATION
# ============================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(
        0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_width * inter_height
    )

    area_a = (
        (ax2 - ax1)
        * (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1)
        * (by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# RISK ENGINE
# ============================================================

risk_score = 10

risk_reason = (
    "No immediate visual proximity concern."
)

highest_iou = 0.0

closest_normalized_distance = 999.0


# ============================================================
# VEHICLE-TO-VEHICLE ANALYSIS
# ============================================================

if len(detections) >= 2:

    for i in range(len(detections)):

        for j in range(
            i + 1,
            len(detections)
        ):

            d1 = detections[i]
            d2 = detections[j]

            # ------------------------------------------------
            # CENTER DISTANCE
            # ------------------------------------------------

            dx = (
                d1["center"][0]
                - d2["center"][0]
            )

            dy = (
                d1["center"][1]
                - d2["center"][1]
            )

            pixel_distance = math.sqrt(
                dx * dx + dy * dy
            )

            diagonal = math.sqrt(
                image_w ** 2
                + image_h ** 2
            )

            normalized_distance = (
                pixel_distance
                / diagonal
            )

            closest_normalized_distance = min(
                closest_normalized_distance,
                normalized_distance
            )

            # ------------------------------------------------
            # BOUNDING BOX OVERLAP
            # ------------------------------------------------

            iou = calculate_iou(
                d1["bbox"],
                d2["bbox"]
            )

            highest_iou = max(
                highest_iou,
                iou
            )

            # ------------------------------------------------
            # BASE RISK SCORE
            # ------------------------------------------------

            pair_score = 10

            # Strong overlap
            if iou > 0.25:

                pair_score = max(
                    pair_score,
                    95
                )

            # Moderate overlap
            elif iou > 0.10:

                pair_score = max(
                    pair_score,
                    75
                )

            # ------------------------------------------------
            # NORMALIZED PROXIMITY
            # ------------------------------------------------

            if normalized_distance < 0.10:

                pair_score = max(
                    pair_score,
                    85
                )

            elif normalized_distance < 0.18:

                pair_score = max(
                    pair_score,
                    65
                )

            elif normalized_distance < 0.28:

                pair_score = max(
                    pair_score,
                    35
                )

            # ------------------------------------------------
            # OBJECT SCALE
            # ------------------------------------------------

            largest_vehicle = max(
                d1["area_ratio"],
                d2["area_ratio"]
            )

            if largest_vehicle > 0.20:

                pair_score += 5

            # ------------------------------------------------
            # VISIBILITY MODIFIER
            # ------------------------------------------------

            if visibility_state == "REDUCED":

                pair_score += 10

            elif visibility_state == "LOW":

                pair_score += 20

            # Limit score
            pair_score = min(
                pair_score,
                100
            )

            risk_score = max(
                risk_score,
                pair_score
            )


# ============================================================
# LOW-VISIBILITY SAFETY EFFECT
# ============================================================

if visibility_state == "LOW":

    # Low visibility itself requires
    # heightened operator awareness.

    risk_score = max(
        risk_score,
        30
    )

    if len(detections) == 0:

        risk_reason = (
            "Low visibility detected with no "
            "reliable vehicle perception."
        )


elif visibility_state == "REDUCED":

    risk_score = max(
        risk_score,
        20
    )


# ============================================================
# FINAL RISK CLASSIFICATION
# ============================================================

if risk_score >= 80:

    risk_level = "CRITICAL"

    risk_reason = (
        "High vehicle proximity detected under "
        f"{visibility_state.lower()} visibility."
    )


elif risk_score >= 60:

    risk_level = "ELEVATED"

    risk_reason = (
        "Vehicle proximity requires increased "
        f"awareness under "
        f"{visibility_state.lower()} visibility."
    )


elif risk_score >= 30:

    risk_level = "WARNING"

    if visibility_state == "LOW":

        risk_reason = (
            "Low visibility requires heightened "
            "operator awareness."
        )

    else:

        risk_reason = (
            "Multiple vehicles show moderate "
            "visual proximity."
        )


else:

    risk_level = "NORMAL"


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()

print("================================================")
print("              MINEGUARD V3")
print("       VISIBILITY-AWARE RISK ENGINE")
print("================================================")

print(
    f"Image       : {IMAGE_PATH}"
)

print(
    f"Image size  : "
    f"{image_w} x {image_h}"
)

print("-----------------------------------------------")

print(
    f"Visibility  : "
    f"{visibility_state}"
)

print(
    f"Contrast    : "
    f"{contrast:.1f}"
)

print(
    f"Sharpness   : "
    f"{laplacian_variance:.1f}"
)

print("-----------------------------------------------")

print(
    f"Vehicles    : "
    f"{len(detections)}"
)

for i, d in enumerate(
    detections,
    start=1
):

    print(
        f"Vehicle {i}: "
        f"{d['class']} | "
        f"confidence="
        f"{d['confidence']:.2f} | "
        f"center=("
        f"{int(d['center'][0])}, "
        f"{int(d['center'][1])})"
    )


print("-----------------------------------------------")

if len(detections) >= 2:

    print(
        f"Closest normalized distance : "
        f"{closest_normalized_distance:.3f}"
    )

    print(
        f"Highest bounding-box IoU     : "
        f"{highest_iou:.3f}"
    )


print("-----------------------------------------------")

print(
    f"RISK SCORE  : "
    f"{risk_score}/100"
)

print(
    f"RISK LEVEL  : "
    f"{risk_level}"
)

print(
    f"REASON      : "
    f"{risk_reason}"
)

print("-----------------------------------------------")


if risk_level == "CRITICAL":

    print(
        "ALERT       : "
        "CRITICAL PROXIMITY"
    )

elif risk_level == "ELEVATED":

    print(
        "ALERT       : "
        "ELEVATED RISK"
    )

elif risk_level == "WARNING":

    print(
        "ALERT       : "
        "HEIGHTENED AWARENESS"
    )

else:

    print(
        "STATUS      : "
        "NORMAL"
    )


# ============================================================
# EVENT LOGGING
# ============================================================

event = {

    "timestamp":
        datetime.now().isoformat(
            timespec="seconds"
        ),

    "image":
        IMAGE_PATH,

    "vehicles_detected":
        len(detections),

    "visibility":
        visibility_state,

    "visibility_level":
        visibility_level,

    "contrast":
        round(contrast, 2),

    "sharpness":
        round(laplacian_variance, 2),

    "risk_score":
        risk_score,

    "risk_level":
        risk_level,

    "risk_reason":
        risk_reason,

    "vehicles": [

        {
            "class":
                d["class"],

            "confidence":
                round(
                    d["confidence"],
                    3
                ),

            "center": [

                int(
                    d["center"][0]
                ),

                int(
                    d["center"][1]
                )
            ],

            "bbox": [

                int(
                    d["bbox"][0]
                ),

                int(
                    d["bbox"][1]
                ),

                int(
                    d["bbox"][2]
                ),

                int(
                    d["bbox"][3]
                )
            ]
        }

        for d in detections
    ]
}


# ============================================================
# SAVE EVENT AS JSONL
# ============================================================

event_file = (
    EVENT_DIR
    / "events.jsonl"
)

with open(
    event_file,
    "a"
) as f:

    f.write(
        json.dumps(event)
        + "\n"
    )


print(
    f"EVENT LOG   : "
    f"{event_file}"
)

print("================================================")
print()