

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# SETTINGS TO TUNE FOR YOUR CAN
# ============================================================

# Joint is the circular boundary detected by HoughCircles.
# Inspect a narrow band around that joint.
RING_INNER_OFFSET = -38      # pixels inside detected circle
RING_OUTER_OFFSET = 55       # pixels outside detected circle

# Polar sampling resolution.
ANGLE_SAMPLES = 2048

# Local smoothing of the polar image.
SIGMA_ANGLE = 4.0
SIGMA_RADIUS = 1.5

# Strong seed: must be a clearly bright local anomaly.
STRONG_Z = 2.8
STRONG_RESIDUAL = 3.0

# Weak pixels are allowed to grow around a strong seed.
GROW_Z = 0.9
GROW_RESIDUAL = 1.0

# Connected component filtering in POLAR coordinates.
MIN_POLAR_AREA = 180
MIN_RADIAL_WIDTH = 8
MAX_RADIAL_WIDTH = 55
MIN_ANGULAR_WIDTH_DEG = 3.0
MAX_ANGULAR_WIDTH_DEG = 35.0

# Mean positive brightness gain required inside candidate.
MIN_MEAN_GAIN = 2.0

# Reject candidates occupying too much of the ring.
MAX_RING_FRACTION = 0.12

# Blue pen/rectangle annotation exclusion for development images.
# Set to False for normal raw production images if desired.
IGNORE_BLUE_MARKUP = True


# ============================================================
# IMAGE / GEOMETRY
# ============================================================

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def detect_main_circle(img: np.ndarray):
    """
    Detect the strongest large circle near the image centre.
    Returns (cx, cy, radius).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=min_dim * 0.35,
        param1=110,
        param2=38,
        minRadius=int(min_dim * 0.20),
        maxRadius=int(min_dim * 0.42),
    )

    if circles is None:
        raise RuntimeError(
            "Could not detect the main bottom circle. "
            "Try lowering HOUGH param2 or use a fixed circle."
        )

    circles = np.round(circles[0]).astype(int)

    # Image centre preference + large-circle preference.
    cx0, cy0 = w / 2.0, h / 2.0
    best = None
    best_score = 1e9

    for cx, cy, r in circles:
        centre_error = math.hypot(cx - cx0, cy - cy0) / min_dim
        radius_error = abs(r - min_dim * 0.27) / min_dim
        score = centre_error + 0.25 * radius_error

        if score < best_score:
            best_score = score
            best = (int(cx), int(cy), int(r))

    return best


def blue_annotation_mask(img: np.ndarray):
    """
    Remove blue hand-drawn annotations if present in a development image.
    """
    if not IGNORE_BLUE_MARKUP:
        return np.ones(img.shape[:2], np.uint8) * 255

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # OpenCV H range = 0..179.
    lower = np.array([90, 90, 70], np.uint8)
    upper = np.array([135, 255, 255], np.uint8)

    blue = cv2.inRange(hsv, lower, upper)

    # Dilate slightly so anti-aliased blue edges are removed too.
    blue = cv2.dilate(
        blue,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    valid = np.where(blue > 0, 0, 255).astype(np.uint8)
    return valid


# ============================================================
# POLAR TRANSFORM
# ============================================================

def build_polar_ring(img: np.ndarray, circle):
    """
    Returns polar L-channel, a-channel, b-channel, validity map and mapping
    arrays.

    Rows     = radius
    Columns  = angle
    """
    cx, cy, r = circle

    r0 = max(1, int(r + RING_INNER_OFFSET))
    r1 = int(r + RING_OUTER_OFFSET)

    if r1 <= r0:
        raise ValueError("Invalid ring offsets.")

    radius_values = np.arange(r0, r1, dtype=np.float32)
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        ANGLE_SAMPLES,
        endpoint=False,
        dtype=np.float32,
    )

    R, A = np.meshgrid(radius_values, theta, indexing="ij")

    map_x = (cx + R * np.cos(A)).astype(np.float32)
    map_y = (cy + R * np.sin(A)).astype(np.float32)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    A_ch = lab[:, :, 1].astype(np.float32)
    B_ch = lab[:, :, 2].astype(np.float32)

    polar_L = cv2.remap(
        L, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    polar_A = cv2.remap(
        A_ch, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    polar_B = cv2.remap(
        B_ch, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    valid = np.ones_like(polar_L, dtype=np.uint8) * 255

    return (
        polar_L,
        polar_A,
        polar_B,
        valid,
        map_x,
        map_y,
        radius_values,
    )


# ============================================================
# PAINT ANOMALY SCORE
# ============================================================

def circular_rolling_median(arr: np.ndarray, window: int):
    """
    Median around the ANGLE axis with wrap-around.
    """
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1

    pad = window // 2
    extended = np.concatenate([arr[:, -pad:], arr, arr[:, :pad]], axis=1)

    # medianBlur works efficiently for uint8; for float use percentile-based
    # rolling median manually in blocks.
    out = np.empty_like(arr)

    for x in range(arr.shape[1]):
        region = extended[:, x:x + window]
        out[:, x] = np.median(region, axis=1)

    return out


def compute_anomaly_score(polar_L: np.ndarray):
    """
    Remove the normal circular/radial background.

    For each radius:
       baseline = robust median brightness across all angles

    Then detect only positive deviations.
    """
    # Light smoothing without destroying the defect.
    smoothed = cv2.GaussianBlur(
        polar_L,
        (0, 0),
        sigmaX=SIGMA_ANGLE,
        sigmaY=SIGMA_RADIUS,
    )

    # Circular median over angle suppresses a localized paint blob while
    # preserving the normal ring brightness at that radius.
    local_bg = circular_rolling_median(
        smoothed,
        window=81,
    )

    residual = smoothed - local_bg

    # Robust scale per radius.
    med = np.median(residual, axis=1, keepdims=True)
    mad = np.median(
        np.abs(residual - med),
        axis=1,
        keepdims=True,
    )

    scale = 1.4826 * mad + 0.75

    z = (residual - med) / scale

    return residual, z


def get_candidate_mask(
    polar_L: np.ndarray,
    polar_A: np.ndarray,
    polar_B: np.ndarray,
):
    residual, z = compute_anomaly_score(polar_L)

    # Strong seeds.
    seeds = (
        (z >= STRONG_Z) &
        (residual >= STRONG_RESIDUAL)
    ).astype(np.uint8)

    # Weak candidate pixels surrounding a seed.
    weak = (
        (z >= GROW_Z) &
        (residual >= GROW_RESIDUAL)
    ).astype(np.uint8)

    # White paint should be close to neutral. This is a weak secondary
    # condition; it is not used alone because the can is already gray.
    chroma = np.sqrt(
        (polar_A - 128.0) ** 2 +
        (polar_B - 128.0) ** 2
    )

    chroma_med = np.median(chroma)
    # Allow normal gray surfaces but reject strongly coloured pixels.
    neutral = (chroma <= chroma_med + 18.0)

    weak = (weak & neutral).astype(np.uint8)

    # Clean weak map.
    weak = cv2.morphologyEx(
        weak,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 11)
        ),
    )

    # Connected strong seeds.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (seeds * 255).astype(np.uint8),
        8,
    )

    grown = np.zeros_like(seeds)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 4:
            continue

        seed = (labels == i).astype(np.uint8)

        # Grow moderately in angle and radius.
        seed_d = cv2.dilate(
            seed,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (17, 11),
            ),
            iterations=1,
        )

        region = seed_d & weak

        region = cv2.morphologyEx(
            region,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (11, 7),
            ),
        )

        grown |= region

    # Final cleanup.
    grown = cv2.morphologyEx(
        grown,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )

    return grown, residual, z


# ============================================================
# CANDIDATE FILTERING
# ============================================================

def filter_polar_components(mask: np.ndarray, residual: np.ndarray):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask * 255).astype(np.uint8),
        8,
    )

    accepted = np.zeros_like(mask)
    details = []

    ring_height = mask.shape[0]
    angle_width = 360.0 / ANGLE_SAMPLES

    for i in range(1, num):

        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        angular_width = w * angle_width
        radial_width = h

        if area < MIN_POLAR_AREA:
            continue

        if radial_width < MIN_RADIAL_WIDTH:
            continue

        if radial_width > MAX_RADIAL_WIDTH:
            continue

        if angular_width < MIN_ANGULAR_WIDTH_DEG:
            continue

        if angular_width > MAX_ANGULAR_WIDTH_DEG:
            continue

        region = labels == i
        values = residual[region]

        mean_gain = float(np.mean(values))

        if mean_gain < MIN_MEAN_GAIN:
            continue

        # Reject candidate covering an excessive portion of the whole ring.
        ring_fraction = area / float(mask.size)

        if ring_fraction > MAX_RING_FRACTION:
            continue

        accepted[region] = 1

        details.append(
            {
                "area_polar_px": area,
                "angular_width_deg": angular_width,
                "radial_width_px": radial_width,
                "mean_brightness_gain": mean_gain,
                "centroid_angle_deg": float(
                    centroids[i][0] * 360.0 / ANGLE_SAMPLES
                ),
                "centroid_radius_px": float(
                    centroids[i][1]
                ),
            }
        )

    return accepted, details


# ============================================================
# MAP BACK TO ORIGINAL IMAGE
# ============================================================

def map_polar_mask_to_image(
    polar_mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    image_shape,
):
    h, w = image_shape[:2]
    result = np.zeros((h, w), np.uint8)

    ys, xs = np.where(polar_mask > 0)

    if len(xs) == 0:
        return result

    ix = np.rint(map_x[ys, xs]).astype(int)
    iy = np.rint(map_y[ys, xs]).astype(int)

    valid = (
        (ix >= 0) & (ix < w) &
        (iy >= 0) & (iy < h)
    )

    result[iy[valid], ix[valid]] = 255

    # Fill small interpolation gaps.
    result = cv2.dilate(
        result,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        ),
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)
        ),
    )

    return result


def merge_cartesian_components(mask: np.ndarray):
    """
    Final connected-component filtering in image space.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        8,
    )

    out = np.zeros_like(mask)

    components = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        if area < 120:
            continue

        # Avoid huge regions that indicate incorrect circle/illumination.
        if area > 12000:
            continue

        out[labels == i] = 255

        components.append(
            {
                "area_px": area,
                "bbox": [x, y, w, h],
            }
        )

    return out, components


# ============================================================
# VISUALIZATION
# ============================================================

def draw_results(
    img,
    circle,
    final_mask,
    decision,
):
    out = img.copy()

    cx, cy, r = circle

    # Joint circle.
    cv2.circle(
        out,
        (cx, cy),
        r,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )

    # Highlight mask in red.
    red = np.zeros_like(out)
    red[:, :, 2] = 255

    idx = final_mask > 0
    out[idx] = cv2.addWeighted(
        out[idx],
        0.25,
        red[idx],
        0.75,
        0,
    )

    # Draw contours.
    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for cnt in contours:
        cv2.drawContours(
            out,
            [cnt],
            -1,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    color = (
        (0, 0, 255)
        if decision == "REJECTED - EXCESS PAINT"
        else (0, 200, 0)
    )

    cv2.rectangle(out, (20, 20), (690, 90), (0, 0, 0), -1)

    cv2.putText(
        out,
        decision,
        (35, 65),
        cv2.FONT_HERSHEY_DUPLEX,
        1.05,
        color,
        2,
        cv2.LINE_AA,
    )

    return out


def create_polar_visual(polar_L, residual, candidate):
    """
    Human-readable debugging image.
    """
    norm = cv2.normalize(
        polar_L,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    visual = cv2.cvtColor(
        norm,
        cv2.COLOR_GRAY2BGR,
    )

    visual[candidate > 0] = (0, 0, 255)

    return visual


# ============================================================
# MAIN INSPECTION
# ============================================================

def inspect_can(image_path: str):
    img = load_image(image_path)

    # Remove blue annotation from the analysis if present.
    analysis_mask = blue_annotation_mask(img)

    circle = detect_main_circle(img)

    (
        polar_L,
        polar_A,
        polar_B,
        polar_valid,
        map_x,
        map_y,
        radius_values,
    ) = build_polar_ring(img, circle)

    candidate, residual, z = get_candidate_mask(
        polar_L,
        polar_A,
        polar_B,
    )

    # Do not use annotated blue pixels as candidate input.
    # Build a polar blue-valid mask by sampling the image-space validity mask.
    polar_blue_valid = cv2.remap(
        analysis_mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    candidate[polar_blue_valid == 0] = 0

    polar_accepted, polar_details = filter_polar_components(
        candidate,
        residual,
    )

    final_mask = map_polar_mask_to_image(
        polar_accepted,
        map_x,
        map_y,
        img.shape,
    )

    # Do not permit blue markup into final result.
    final_mask[analysis_mask == 0] = 0

    final_mask, image_components = merge_cartesian_components(
        final_mask
    )

    total_area = int(np.count_nonzero(final_mask))

    if total_area > 0:
        decision = "REJECTED - EXCESS PAINT"
    else:
        decision = "PASSED"

    result = draw_results(
        img,
        circle,
        final_mask,
        decision,
    )

    report = {
        "image": str(image_path),
        "decision": decision,
        "circle": {
            "center_x": circle[0],
            "center_y": circle[1],
            "radius_px": circle[2],
        },
        "defect_area_px": total_area,
        "polar_candidates": len(polar_details),
        "image_components": image_components,
        "polar_candidate_details": polar_details,
        "settings": {
            "ring_inner_offset": RING_INNER_OFFSET,
            "ring_outer_offset": RING_OUTER_OFFSET,
            "strong_z": STRONG_Z,
            "strong_residual": STRONG_RESIDUAL,
            "grow_z": GROW_Z,
            "grow_residual": GROW_RESIDUAL,
            "min_polar_area": MIN_POLAR_AREA,
            "min_radial_width": MIN_RADIAL_WIDTH,
            "max_radial_width": MAX_RADIAL_WIDTH,
            "min_angular_width_deg": MIN_ANGULAR_WIDTH_DEG,
            "max_angular_width_deg": MAX_ANGULAR_WIDTH_DEG,
            "min_mean_gain": MIN_MEAN_GAIN,
            "ignore_blue_markup": IGNORE_BLUE_MARKUP,
        },
    }

    return (
        result,
        final_mask,
        create_polar_visual(
            polar_L,
            residual,
            polar_accepted,
        ),
        report,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Detect excessive paint blobs around a can bottom joint."
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="Image file path.",
    )

    args = parser.parse_args()

    if args.image:
        image_path = args.image.strip().strip('"')
    else:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        image_path = filedialog.askopenfilename(
            title="Select can image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        root.destroy()

        if not image_path:
            print("No image selected.")
            return

    result_dir = Path(image_path).parent / "Inspection_Results"
    result_dir.mkdir(exist_ok=True)

    try:
        result, mask, polar_visual, report = inspect_can(
            image_path
        )

        stem = Path(image_path).stem

        result_file = result_dir / f"result_{stem}.jpg"
        mask_file = result_dir / f"mask_{stem}.png"
        polar_file = result_dir / f"polar_{stem}.jpg"
        report_file = result_dir / f"report_{stem}.json"

        cv2.imwrite(str(result_file), result)
        cv2.imwrite(str(mask_file), mask)
        cv2.imwrite(str(polar_file), polar_visual)

        report_file.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 70)
        print("CAN EXCESSIVE PAINT INSPECTION")
        print("=" * 70)
        print(f"Image              : {Path(image_path).name}")
        print(f"Circle center      : ({report['circle']['center_x']}, "
              f"{report['circle']['center_y']})")
        print(f"Circle radius      : {report['circle']['radius_px']} px")
        print(f"Polar candidates   : {report['polar_candidates']}")
        print(f"Final defect area  : {report['defect_area_px']} px")
        print(f"FINAL RESULT       : {report['decision']}")
        print("-" * 70)
        print(f"Result             : {result_file}")
        print(f"Mask               : {mask_file}")
        print(f"Polar debug        : {polar_file}")
        print(f"Report             : {report_file}")
        print("=" * 70)

    except Exception as exc:
        print(f"\nInspection error: {exc}")


if __name__ == "__main__":
    main()
