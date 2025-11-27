import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

# Configuration
IMAGES_DIR = "images"
OUTPUT_DIR = "multi_results"
CONFIDENCE_THRESHOLD = 0.55
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Camera and Tag Physical Info
MARKER_SIZE_MM = 53         # printed tag width
FOCAL_LENGTH_MM = 4.25      # camera focal length (mm)
SENSOR_WIDTH_MM = 6.4       # physical width of sensor (mm)

best_w = "strawberry_flower_detection_model/weights/best.pt"
model = YOLO(best_w)
os.makedirs(OUTPUT_DIR, exist_ok=True)

global_tags = {}
global_flowers = []
global_origin = None
mm_per_px_global = None

# detect ArUco tags
def detect_aruco(image):
    """Detect ArUco markers and return dict"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(image)
    result = {}
    if ids is not None:
        ids = ids.flatten()
        for i, cset in enumerate(corners):
            center = np.mean(cset[0], axis=0)
            result[int(ids[i])] = (center, cset[0])
    return result

# compute camera-to-tag distance using optical geometry
def compute_distance_mm(tag_px_width, image_width_px):
    """Estimate camera distance to tag using pinhole camera geometry."""
    pixel_size_mm = SENSOR_WIDTH_MM / image_width_px
    tag_width_on_sensor_mm = tag_px_width * pixel_size_mm
    distance_mm = (FOCAL_LENGTH_MM * MARKER_SIZE_MM) / tag_width_on_sensor_mm
    return distance_mm

def process_image(img_path):
    global global_origin, mm_per_px_global

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    tags = detect_aruco(img)
    if not tags:
        print(f"No ArUco tags detected in {img_path}")
        return

    # use first detected tag for calibration
    first_id = list(tags.keys())[0]
    _, corners = tags[first_id]

    # Compute average tag side length in pixels
    sides = [
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[1] - corners[2]),
        np.linalg.norm(corners[2] - corners[3]),
        np.linalg.norm(corners[3] - corners[0]),
    ]
    tag_px_width = np.mean(sides)

    mm_per_px = MARKER_SIZE_MM / tag_px_width

    mm_per_px_global = mm_per_px

    print(f"\nProcessing: {os.path.basename(img_path)}")

    # establish origin 
    if global_origin is None and 0 in tags:
        global_origin = np.array([0.0, 0.0])
        global_tags[0] = {"center_mm": [0.0, 0.0]}
        print("Tag0 found → set origin (0,0)")

    # find offset from any shared tag
    offset = None
    for tid, (center, _) in tags.items():
        if tid in global_tags and offset is None:
            known_mm = np.array(global_tags[tid]["center_mm"])
            offset = known_mm - (center * mm_per_px)
    if offset is None:
        offset = np.array([0.0, 0.0])

    # record tag positions in mm
    for tid, (center, _) in tags.items():
        pos_mm = center * mm_per_px + offset
        global_tags[tid] = {"center_mm": [float(pos_mm[0]), float(pos_mm[1])]}
        cv2.circle(img, tuple(map(int, center)), 8, (0, 255, 255), -1)
        cv2.putText(
            img,
            f"Tag{tid}",
            tuple(map(int, center + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # run flower detection
    yolo_results = model.predict(img_path, conf=CONFIDENCE_THRESHOLD, imgsz=1280)[0]
    detections = []

    for box in yolo_results.boxes:
        detections.append(
            type(
                "det",
                (),
                {
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "x": float(box.xywh[0][0]),
                    "y": float(box.xywh[0][1]),
                    "width": float(box.xywh[0][2]),
                    "height": float(box.xywh[0][3]),
                },
            )
        )

    local_flowers = []

    for det in detections:
        # Draw ALL detections on the individual image (all stages)
        obj_center = np.array([det.x, det.y])
        x1, y1 = int(det.x - det.width / 2), int(det.y - det.height / 2)
        x2, y2 = int(det.x + det.width / 2), int(det.y + det.height / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{det.class_name} ({det.confidence:.2f})",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Only FULL flowers are added to the merged world map + local path
        if det.class_name == "full":
            rel_mm = obj_center * mm_per_px + offset
            global_flowers.append(
                {
                    "label": det.class_name,
                    "conf": round(float(det.confidence), 2),
                    "center_mm": [
                        round(float(rel_mm[0]), 2),
                        round(float(rel_mm[1]), 2),
                    ],
                    "source": os.path.basename(img_path),
                }
            )
            local_flowers.append(obj_center)

    # draw local path
    if 0 in global_tags and local_flowers:
        start_px = tags[0][0] if 0 in tags else local_flowers[0]
        local_flowers.sort(key=lambda a: np.linalg.norm(a - start_px))
        prev = tuple(map(int, start_px))
        for i, a in enumerate(local_flowers, 1):
            ap_px = tuple(map(int, a))
            cv2.line(img, prev, ap_px, (0, 0, 255), 2)
            cv2.putText(
                img,
                f"flower{i}",
                (ap_px[0] + 5, ap_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            prev = ap_px

    out_local = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_local, img)
    print(f"Saved annotated image: {out_local}")


# Process all images
files = [
    f
    for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
files.sort()

for fname in files:
    process_image(os.path.join(IMAGES_DIR, fname))

# Build merged visualization
if not global_tags or not global_flowers:
    print("No data collected, skipping global map.")
    raise SystemExit

min_x = min_y = float("inf")
max_x = max_y = float("-inf")

for tag in global_tags.values():
    x, y = tag["center_mm"]
    min_x, min_y = min(min_x, x), min(min_y, y)
    max_x, max_y = max(max_x, x), max(max_y, y)

for app in global_flowers:
    x, y = app["center_mm"]
    min_x, min_y = min(min_x, x), min(min_y, y)
    max_x, max_y = max(max_x, x), max(max_y, y)

scale = 1 / mm_per_px_global * 0.1
canvas_w = int((max_x - min_x) * scale + 400)
canvas_h = int((max_y - min_y) * scale + 400)
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255


def mm_to_px(pt):
    return int((pt[0] - min_x) * scale + 100), int((pt[1] - min_y) * scale + 100)


# draw tags
for tid, tag in global_tags.items():
    px = mm_to_px(tag["center_mm"])

    if tid == 0:
        color = (255, 0, 0)
        label = "Tag0 / Start Point"
    else:
        color = (0, 200, 0)
        label = f"Tag{tid}"

    cv2.circle(canvas, px, 4, color, -1)
    cv2.putText(
        canvas,
        label,
        (px[0] + 5, px[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )

# draw path & labels for FULL flowers only
start_mm = np.array(global_tags[0]["center_mm"])
flowers_sorted = sorted(
    global_flowers,
    key=lambda a: np.linalg.norm(np.array(a["center_mm"]) - start_mm),
)
prev = mm_to_px(start_mm)

# cluster close flowers in pixel space
CLUSTER_DIST_PX = 12
clusters = []

for a in flowers_sorted:
    pt_mm = np.array(a["center_mm"])
    pt_px = np.array(mm_to_px(pt_mm))

    assigned = False
    for c in clusters:
        if np.linalg.norm(pt_px - c["center_px"]) <= CLUSTER_DIST_PX:
            c["points"].append(pt_px)
            assigned = True
            break

    if not assigned:
        clusters.append(
            {
                "center_mm": pt_mm,
                "center_px": pt_px,
                "points": [pt_px],
            }
        )

# draw each cluster with flower labels
flower_number = 1

for cluster in clusters:
    px = cluster["center_px"]
    count = len(cluster["points"])

    # path line from previous point
    cv2.line(canvas, prev, tuple(px), (0, 0, 255), 2)

    # dot at cluster center
    cv2.circle(canvas, tuple(px), 5, (0, 0, 255), -1)

    # labels for each flower in this cluster
    for j in range(count):
        label_x = int(px[0] + 10)
        label_y = int(px[1] - 10 - j * 15)

        cv2.putText(
            canvas,
            f"flower{flower_number}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )

        flower_number += 1

    prev = tuple(px)

# save outputs
out_img = os.path.join(OUTPUT_DIR, "merged_world_map.jpg")
out_json = os.path.join(OUTPUT_DIR, "merged_world_map.json")
cv2.imwrite(out_img, canvas)
with open(out_json, "w") as f:
    json.dump({"tags": global_tags, "flowers": global_flowers}, f, indent=4)

print(f"\nSaved merged world map: {out_img}")
print(f"Saved coordinates JSON: {out_json}")
