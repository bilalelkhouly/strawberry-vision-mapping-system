import json
import math
import os

# ======== CONFIG ========
JSON_PATH = "multi_results/merged_world_map.json"

# Drift compensation factors
DRIFT_DEFAULT = 0.85   # applies to forward/backward motion (Y-axis)
DRIFT_RIGHT = 0.5      # applies to rightward motion (Δx < 0)
DRIFT_LEFT = 1.15      # applies to leftward motion (Δx > 0)
# =========================

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ JSON file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

def direction_from_deltas(dx, dy):
    """Return readable motion direction (flipped X-axis for drone view)."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "No movement"

    dir_x = ""
    dir_y = ""

    # ⚠️ Flipped X-axis: +X = LEFT, -X = RIGHT
    if dx > 0:
        dir_x = "← Left"
    elif dx < 0:
        dir_x = "→ Right"

    # Normal Y-axis: +Y = Forward
    if dy > 0:
        dir_y = "↑ Forward"
    elif dy < 0:
        dir_y = "↓ Backward"

    return f"{dir_x} {dir_y}".strip() or "No movement"

# --- MAIN EXECUTION ---
data = load_data(JSON_PATH)
tags = data.get("tags", {})
apples = data.get("apples", [])

if "0" not in tags:
    raise ValueError("❌ Tag 0 (origin) not found in JSON!")

start = tags["0"]["center_mm"]

# Sort apples by distance from Tag0
apples_sorted = sorted(apples, key=lambda a: math.dist(a["center_mm"], start))

# Build path: Tag0 → P1 → P2 → ...
path = [("Tag0", start)] + [(f"P{i+1}", a["center_mm"]) for i, a in enumerate(apples_sorted)]

print("=" * 75)
print("PATH SEGMENT REPORT (in meters, high precision, directional drift applied)")
print("=" * 75)

total_m_raw = 0.0
total_m_drift = 0.0

for i in range(len(path) - 1):
    name1, (x1_mm, y1_mm) = path[i]
    name2, (x2_mm, y2_mm) = path[i + 1]

    # Convert to meters
    x1, y1 = x1_mm / 1000.0, y1_mm / 1000.0
    x2, y2 = x2_mm / 1000.0, y2_mm / 1000.0

    # Raw deltas
    dx = x2 - x1
    dy = y2 - y1
    dist_raw = math.sqrt(dx**2 + dy**2)
    total_m_raw += dist_raw

    # --- Asymmetric drift compensation ---
    if dx < 0:  # Rightward (negative X)
        dx_adj = dx * DRIFT_RIGHT
        drift_label = "Right drift (0.5×)"
    elif dx > 0:  # Leftward (positive X)
        dx_adj = dx * DRIFT_LEFT
        drift_label = "Left boosted (1.15×)"
    else:
        dx_adj = dx
        drift_label = "No horizontal drift"

    # Apply same factor for vertical (Y-axis)
    dy_adj = dy * DRIFT_DEFAULT
    dist_adj = math.sqrt(dx_adj**2 + dy_adj**2)
    total_m_drift += dist_adj

    direction = direction_from_deltas(dx, dy)

    print(f"\n🔹 Segment {i+1}: {name1} → {name2}")
    print(f"   Δx = {dx:+.6f} m  |  Adjusted: {dx_adj:+.6f} m  ({drift_label})")
    print(f"   Δy = {dy:+.6f} m  |  Adjusted: {dy_adj:+.6f} m  (Vertical drift 0.85×)")
    print(f"   Distance = {dist_raw:.6f} m")
    print(f"   Adjusted Distance = {dist_adj:.6f} m")
    print(f"   Direction: {direction}")

print("\n" + "=" * 75)
print(f"📏 TOTAL PATH LENGTH (raw): {total_m_raw:.6f} m")
print(f"📏 TOTAL PATH LENGTH (after directional drift): {total_m_drift:.6f} m")
print("=" * 75)

print("\nAxis Reference (Drone View):")
print("   +X → Left")
print("   -X → Right")
print("   +Y → Forward")
print("   -Y → Backward")
print("\n✅ Done! Left movement is now boosted (1.15×) to fight right drift,")
print("   and rightward motion is reduced to 0.5× for precision correction.")
