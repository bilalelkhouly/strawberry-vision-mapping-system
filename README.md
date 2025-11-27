# Strawberry Vision Mapping System

A lightweight computer vision pipeline for detecting strawberry flowers using custom YOLOv8 model and mapping their real-world positions using ArUCo markers. The system merges multiple images into one unified global coordinate map for use in robotics, navigation, and autonomous pollination.

---

## Overview

This project performs:

- Strawberry flower detection using a custom YOLOv8 model  
- ArUCo marker detection for coordinate scaling  
- Pixel → millimeter conversion using optical geometry  
- Global alignment of detections across multiple images  
- World-map generation with flower positions and navigation paths  
- Export of JSON + annotated images  

---

## Installation

Install dependencies with:

    pip install -r requirements.txt

Or manually:

    pip install ultralytics opencv-python numpy

---

## ▶How to Run

1. Place images in the `images/` folder.  
2. Ensure the YOLO model weights are located at:  
   `strawberry_flower_detection_model/weights/best.pt`  
3. Run the script:

    python strawberry_flower_mapping.py

---

## Outputs

The script generates:

- Annotated images showing flower detections and ArUCo tag positions (saved in `multi_results/`)  
- Global merged world map: `merged_world_map.jpg`  
- Coordinate dataset (tags + flowers): `merged_world_map.json`  

---

## Description

- ArUCo tags provide metric scaling based on their known physical size.  
- Detected `"full"` strawberry flowers are converted from pixel coordinates to millimeters.  
- Shared ArUCo tag IDs allow merging detections from multiple images into a single global coordinate frame.  
- A navigation-style path is drawn from Tag 0 (origin) to each detected flower in both local and global views.  
- Close flowers in the global map are clustered to avoid overlapping labels when visualized.

