import cv2
import numpy as np
import os
import sys
import math

# ================================================================
# CAN EXCESSIVE-PAINT DETECTOR - OpenCV only
# ================================================================
# Purpose:
# Detect a bright/white excessive-paint patch concentrated at the
# circular joint between the can bottom and the can wall.
#
# This version intentionally does NOT search for generic scratches,
# dark spots, or arbitrary defects.
#
# Strategy:
#   1. Detect the circular bottom/joint.
#   2. Build a narrow annular inspection band around that joint.
#   3. Convert the image to LAB and use the L (lightness) channel.
#   4. Unwrap that circular band into angle/radius coordinates.
#   5. Find a LOCAL bright angular sector using a Difference-of-
#      Gaussian profile. This suppresses continuous circular lines
#      and broad illumination gradients.
#   6. Inside the suspicious sector, keep only pixels that are
#      brighter than the local paint surface.
#   7. Clean the result and return OK / REJECTED.
#
# Requires:
#   pip install opencv-python numpy
# ================================================================


# -------------------- MAIN TUNING PARAMETERS --------------------

# If Hough detection is unreliable on your raw dataset, set a fixed
# circle here as: (center_x, center_y, radius)
# Example from the supplied image is approximately (945, 746, 425).
MANUAL_CIRCLE = None

# Hough circle search range as a fraction of image's smaller dimension.
MIN_RADIUS_FRACTION = 0.22
MAX_RADIUS_FRACTION = 0.49

# Target joint band around the detected circular joint.
# Negative = slightly inside the floor, positive = toward the can wall.
JOINT_INNER_OFFSET = -10
JOINT_OUTER_OFFSET = 10

# Polar sampling.
ANGLE_SAMPLES = 720       # 0.5 degree per sample
RADIAL_SAMPLES = 60

# Local bright-sector detection.
# 4 samples ~ 2 degrees; 18 samples ~ 9 degrees.
ANGLE_SMALL_SIGMA = 4
ANGLE_LARGE_SIGMA = 18
ANGLE_THRESHOLD_PERCENTILE = 98.0
MIN_ANGLE_WIDTH_DEG = 1.5
MAX_ANGLE_WIDTH_DEG = 35.0

# Brightness confirmation inside suspicious angular sectors.
MIN_BRIGHTNESS_DELTA = 5.0
LOCAL_BACKGROUND_SIGMA = 7.0
MIN_LOCAL_HIGHLIGHT = 1.5

# Morphological cleanup.
OPEN_KERNEL = 3
CLOSE_KERNEL = 7

# Defect size limits.
MIN_DEFECT_AREA = 30
MAX_DEFECT_AREA = 2500

# Optional: remove hand-drawn blue/green annotation from test images.
# WARNING: if the annotation covers the actual defect, the original
# defect pixels are no longer available and cannot be recovered.
IGNORE_BLUE_MARKUP = False


# -------------------- GENERAL FUNCTIONS --------------------

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    return img


def detect_joint_circle(gray):
    """Detect the circular floor/wall joint."""
    h, w = gray.shape
    small = min(h, w)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(small * 0.22),
        param1=120,
        param2=40,
        minRadius=int(small * MIN_RADIUS_FRACTION),
        maxRadius=int(small * MAX_RADIUS_FRACTION),
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Prefer a circle near the image centre and with a plausible size.
    cx0, cy0 = w / 2.0, h / 2.0
    best = None
    best_score = float("inf")

    for cx, cy, r in circles:
        center_error = math.hypot(cx - cx0, cy - cy0) / small
        size_error = abs((2 * r) - 0.70 * small) / small
        score = center_error + 0.35 * size_error
        if score < best_score:
            best_score = score
            best = (int(cx), int(cy), int(r))

    return best


def circular_gaussian(signal, sigma):
    """Gaussian blur with wrap-around for circular angle data."""
    signal = np.asarray(signal, dtype=np.float32)
    extended = np.concatenate([signal, signal, signal])
    blurred = cv2.GaussianBlur(
        extended.reshape(1, -1),
        (0, 0),
        sigmaX=float(sigma),
    )[0]
    n = len(signal)
    return blurred[n:2 * n]


def build_polar_maps(cx, cy, r_inner, r_outer, angle_samples, radial_samples):
    angles = np.linspace(0.0, 2.0 * np.pi, angle_samples, endpoint=False)
    radii = np.linspace(r_inner, r_outer, radial_samples)

    rr, tt = np.meshgrid(radii, angles, indexing="ij")
    map_x = (cx + rr * np.cos(tt)).astype(np.float32)
    map_y = (cy + rr * np.sin(tt)).astype(np.float32)

    return map_x, map_y, angles, radii


def detect_blue_markup(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # OpenCV blue range.
    mask = cv2.inRange(
        hsv,
        np.array([90, 100, 60], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8),
    )
    return mask > 0


def make_sector_mask(
    angle_score,
    threshold,
    min_width_deg,
    max_width_deg,
):
    """Find local angular sectors that are bright anomalies."""
    flagged = (angle_score >= threshold).astype(np.uint8)

    # Connect neighbouring high-angle samples.
    close_kernel = np.ones((1, 15), np.uint8)
    open_kernel = np.ones((1, 5), np.uint8)
    row = flagged.reshape(1, -1)
    row = cv2.morphologyEx(row, cv2.MORPH_CLOSE, close_kernel)[0]
    row = cv2.morphologyEx(row.reshape(1, -1), cv2.MORPH_OPEN, open_kernel)[0]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        row.reshape(1, -1), 8
    )

    result = np.zeros_like(flagged, dtype=bool)
    sample_deg = 360.0 / len(flagged)

    for i in range(1, n):
        width_samples = int(stats[i, cv2.CC_STAT_WIDTH])
        width_deg = width_samples * sample_deg

        if min_width_deg <= width_deg <= max_width_deg:
            result[labels[0] == i] = True

    return result


def inspect_can(image_path):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)

    # 1) Locate the circular joint.
    if MANUAL_CIRCLE is not None:
        circle = MANUAL_CIRCLE
    else:
        circle = detect_joint_circle(gray)

    if circle is None:
        raise RuntimeError(
            "Could not detect the can bottom/joint circle. "
            "Try MANUAL_CIRCLE = (cx, cy, radius)."
        )

    cx, cy, radius = circle

    # 2) Define a thin band centered on the circular joint.
    r_inner = max(1, radius + JOINT_INNER_OFFSET)
    r_outer = radius + JOINT_OUTER_OFFSET

    map_x, map_y, angles, radii = build_polar_maps(
        cx,
        cy,
        r_inner,
        r_outer,
        ANGLE_SAMPLES,
        RADIAL_SAMPLES,
    )

    # 3) Convert the circular joint to polar coordinates.
    polar_L = cv2.remap(
        lightness,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # 4) Optional test-image markup masking.
    invalid_polar = np.zeros_like(polar_L, dtype=bool)
    if IGNORE_BLUE_MARKUP:
        blue = detect_blue_markup(img).astype(np.uint8)
        invalid_polar = cv2.remap(
            blue,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        polar_L = polar_L.copy()
        polar_L[invalid_polar] = np.nan

    # 5) Build an angle profile.
    # The 85th percentile is useful because excessive paint may occupy only
    # part of the radial band instead of the entire width of the joint.
    angle_profile = np.nanpercentile(polar_L, 85, axis=0)

    # Local-vs-broad angular contrast.
    small = circular_gaussian(angle_profile, ANGLE_SMALL_SIGMA)
    large = circular_gaussian(angle_profile, ANGLE_LARGE_SIGMA)
    angle_score = small - large

    valid_score = angle_score[np.isfinite(angle_score)]
    threshold = np.percentile(valid_score, ANGLE_THRESHOLD_PERCENTILE)

    suspicious_angles = make_sector_mask(
        angle_score,
        threshold,
        MIN_ANGLE_WIDTH_DEG,
        MAX_ANGLE_WIDTH_DEG,
    )

    # 6) Convert suspicious angles back to Cartesian coordinates.
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    joint_mask = (
        (distance >= r_inner)
        & (distance <= r_outer)
    )

    angle_image = (np.arctan2(yy - cy, xx - cx) % (2 * np.pi))
    angle_index = (
        angle_image / (2 * np.pi) * ANGLE_SAMPLES
    ).astype(np.int32) % ANGLE_SAMPLES

    suspicious_cartesian = suspicious_angles[angle_index]

    candidate_roi = joint_mask & suspicious_cartesian

    if IGNORE_BLUE_MARKUP:
        candidate_roi &= ~detect_blue_markup(img)

    # 7) Brightness test. Excessive paint is expected to be MORE WHITE,
    # therefore we reject dark anomalies rather than searching for all
    # contrast changes.
    roi_values = lightness[candidate_roi]
    if roi_values.size < 20:
        # No usable candidate sector.
        defect_mask = np.zeros_like(gray, dtype=np.uint8)
        candidates = []
    else:
        sector_median = float(np.median(roi_values))
        bright_threshold = sector_median + MIN_BRIGHTNESS_DELTA

        # Local high-pass term suppresses broad illumination gradients.
        local_background = cv2.GaussianBlur(
            lightness,
            (0, 0),
            sigmaX=LOCAL_BACKGROUND_SIGMA,
        )
        local_highlight = lightness - local_background

        candidate = (
            candidate_roi
            & (lightness >= bright_threshold)
            & (local_highlight >= MIN_LOCAL_HIGHLIGHT)
        )

        defect_mask = np.zeros_like(gray, dtype=np.uint8)
        defect_mask[candidate] = 255

        # 8) Clean the candidate region.
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (OPEN_KERNEL, OPEN_KERNEL),
        )
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CLOSE_KERNEL, CLOSE_KERNEL),
        )

        defect_mask = cv2.morphologyEx(
            defect_mask,
            cv2.MORPH_OPEN,
            kernel_open,
        )
        defect_mask = cv2.morphologyEx(
            defect_mask,
            cv2.MORPH_CLOSE,
            kernel_close,
        )

        # 9) Connected-component filtering.
        filtered = np.zeros_like(defect_mask)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            defect_mask,
            8,
        )

        candidates = []

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_DEFECT_AREA or area > MAX_DEFECT_AREA:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Centroid must stay near the joint.
            cX, cY = centroids[i]
            c_radius = math.hypot(float(cX) - cx, float(cY) - cy)
            if not (r_inner - 3 <= c_radius <= r_outer + 5):
                continue

            # Very thin noise is not an excessive-paint patch.
            aspect = max(ww, hh) / max(1, min(ww, hh))
            if aspect > 10:
                continue

            component_pixels = labels == i
            mean_lightness = float(np.mean(lightness[component_pixels]))
            mean_highlight = float(np.mean(local_highlight[component_pixels]))

            filtered[component_pixels] = 255
            candidates.append({
                "area_px": area,
                "bbox": [x, y, ww, hh],
                "center": [int(cX), int(cY)],
                "mean_lightness": round(mean_lightness, 2),
                "mean_local_highlight": round(mean_highlight, 2),
                "radius_from_joint_center": round(c_radius, 2),
            })

        defect_mask = filtered

    # 10) Final visual output.
    output = img.copy()

    # Draw detected joint band.
    cv2.circle(
        output,
        (cx, cy),
        int(r_inner),
        (0, 255, 255),
        2,
    )
    cv2.circle(
        output,
        (cx, cy),
        int(r_outer),
        (0, 255, 255),
        2,
    )

    # Draw defect contours.
    contours, _ = cv2.findContours(
        defect_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        cv2.drawContours(
            output,
            [contour],
            -1,
            (0, 0, 255),
            3,
        )

        M = cv2.moments(contour)
        if M["m00"] != 0:
            px = int(M["m10"] / M["m00"])
            py = int(M["m01"] / M["m00"])
            cv2.circle(output, (px, py), 5, (0, 0, 255), -1)

    rejected = len(candidates) > 0
    status = "REJECTED - EXCESS PAINT" if rejected else "PASSED"
    status_color = (0, 0, 255) if rejected else (0, 200, 0)

    cv2.putText(
        output,
        f"Status: {status}",
        (25, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Excess paint regions: {len(candidates)}",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Save result next to input image.
    output_dir = os.path.join(os.path.dirname(image_path), "Inspection_Results")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"result_{filename}")
    cv2.imwrite(output_path, output)

    # Also save the binary defect mask.
    mask_path = os.path.join(output_dir, f"mask_{filename}")
    cv2.imwrite(mask_path, defect_mask)

    # Save JSON-like text report.
    report_path = os.path.join(
        output_dir,
        os.path.splitext(filename)[0] + "_report.txt",
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CAN EXCESSIVE-PAINT INSPECTION REPORT\n")
        f.write("=====================================\n")
        f.write(f"Image: {filename}\n")
        f.write(f"Circle center: ({cx}, {cy})\n")
        f.write(f"Circle radius: {radius}\n")
        f.write(f"Joint band: {r_inner:.1f} to {r_outer:.1f} px\n")
        f.write(f"Angle threshold: {threshold:.3f}\n")
        f.write(f"Excess paint regions: {len(candidates)}\n")
        f.write(f"Decision: {status}\n\n")
        for idx, c in enumerate(candidates, 1):
            f.write(f"Region {idx}: {c}\n")

    analytics = {
        "status": status,
        "defect_found": rejected,
        "total_defects": len(candidates),
        "circle": {
            "center_x": cx,
            "center_y": cy,
            "radius": radius,
        },
        "joint_band": {
            "inner_radius": r_inner,
            "outer_radius": r_outer,
        },
        "candidates": candidates,
        "output_file": output_path,
        "mask_file": mask_path,
        "report_file": report_path,
    }

    return output, rejected, analytics


def main():
    # If a path is supplied, process it directly.
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        result_visual, is_defective, data = inspect_can(image_path)
        print("=" * 65)
        print("CAN EXCESSIVE-PAINT INSPECTION")
        print("=" * 65)
        print(f"Image          : {os.path.basename(image_path)}")
        print(f"Joint circle   : {data['circle']}")
        print(f"Joint band     : {data['joint_band']}")
        print(f"Excess regions : {data['total_defects']}")
        print(f"Decision       : {data['status']}")
        print(f"Result         : {data['output_file']}")
        print(f"Mask           : {data['mask_file']}")
        print("=" * 65)
        return

    # Optional interactive terminal loop, similar to your original program.
    base_folder = input("Enter image folder path: ").strip()
    if not base_folder:
        print("No folder supplied.")
        return

    while True:
        user_input = input(
            "Enter image filename (or full path, 'exit' to quit): "
        ).strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Inspection system closed.")
            break

        if not user_input:
            continue

        image_path = user_input
        if not os.path.isabs(image_path):
            image_path = os.path.join(base_folder, image_path)

        if not os.path.exists(image_path):
            print(f"File not found: {image_path}\n")
            continue

        try:
            _, _, data = inspect_can(image_path)
            print("-" * 65)
            print(f"Processed file : {os.path.basename(image_path)}")
            print(f"Status         : {data['status']}")
            print(f"Excess regions : {data['total_defects']}")
            print(f"Saved result   : {data['output_file']}")
            print("-" * 65)
        except Exception as exc:
            print(f"Process error: {exc}\n")


if __name__ == "__main__":
    main()
