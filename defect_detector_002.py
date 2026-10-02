import cv2
import numpy as np
import os
import sys

def inspect_can(image_path):
    """
    Production inspection script optimized to target ONLY excessive paint pools 
    built up along the bottom floor joint groove, ignoring the wall and scratches.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image: {image_path}")
    
    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. DYNAMIC FLOOR CENTER LOCATION
    _, thresh_center = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    M_center = cv2.moments(thresh_center)
    
    if M_center["m00"] != 0:
        center_x = int(M_center["m10"] / M_center["m00"])
        center_y = int(M_center["m01"] / M_center["m00"])
    else:
        center_x, center_y = int(w / 2), int(h / 2)

    # 2. SMOOTH SURFACES & TARGET THE RIM CHANNEL
    # Smooth metallic micro-grain while keeping structural contrast sharp
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Establish a geometric mask tracking ONLY the narrow path around the floor joint rim
    mask_outer = np.zeros((h, w), dtype=np.uint8)
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    
    radius_outer = int(min(h, w) / 4.25)
    radius_inner = int(min(h, w) / 5.25)
    
    cv2.circle(mask_outer, (center_x, center_y), radius_outer, 255, -1)
    cv2.circle(mask_inner, (center_x, center_y), radius_inner, 255, -1)
    
    # This forms a clean circular track over the bottom joint groove, cutting off walls entirely
    joint_groove_track = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

    # 3. SPLIT LOCAL HIGH INTENSITIES (Locating the thick paint puddle)
    # The normal paint layer inside the shadow has a controlled mid-tone gray value.
    # The excessive paint pool builds vertical depth, creating a bright white patch.
    _, bright_paint_mask = cv2.threshold(smoothed, 195, 255, cv2.THRESH_BINARY)
    
    # Isolate bright anomalies that live inside the rim joint channel
    target_zone = cv2.bitwise_and(bright_paint_mask, joint_groove_track)
    
    # Structural cleanup to erase minute loose lighting glare specks
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    target_zone = cv2.morphologyEx(target_zone, cv2.MORPH_OPEN, kernel_clean)

    defects_found = False
    analytics = {"excessive_paint_pools": 0}

    contours, _ = cv2.findContours(target_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Area Filter: Ignores micro pixels, locks onto blocky paint pools matching your blue marks
        if 25 <= area <= 2500:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Geometry Shape Verification:
            # Paint drips have a healthy aspect ratio and thickness compared to stray light tracks
            aspect_ratio = max(w_box, h_box) / min(w_box, h_box)
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            
            # Filter out very thin linear specular streaks
            if aspect_ratio > 4.5 and compactness < 0.05:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # True Excessive Paint defect confirmed
            defects_found = True
            analytics["excessive_paint_pools"] += 1
            
            # Map visual markers
            cv2.drawContours(output_img, [cnt], -1, (255, 0, 0), 2)  # Blue outline marker
            cv2.circle(output_img, (cX, cY), 4, (0, 0, 255), -1)      # Red center dot

    # Display Status Panels
    status_text = "REJECTED" if defects_found else "PASSED"
    status_color = (0, 0, 255) if defects_found else (0, 255, 0)
    cv2.putText(output_img, f"Status: {status_text}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.3, status_color, 2)
    cv2.putText(output_img, f"Paint Pools: {analytics['excessive_paint_pools']}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    return output_img, defects_found, analytics


if __name__ == "__main__":
    base_folder = r"E:\W4Labs\CanInspection"
    output_dir = os.path.join(base_folder, "Inspection_Results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 65)
    print("      🏭 EXCESS PAINT ACCUMULATION INSPECTION SYSTEM 🏭      ")
    print("=" * 65)
    print(f"Base Folder Path: {base_folder}")
    print("Type 'exit' or 'quit' at any prompt to safely close the terminal loop.\n")

    while True:
        user_input = input(" Enter image name to test (e.g., can_test_1.jpg): ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("\nShutting down inspection system safely. Goodbye!")
            break
            
        if not user_input:
            continue
            
        if not os.path.isabs(user_input):
            target_path = os.path.join(base_folder, user_input)
        else:
            target_path = user_input

        if not os.path.exists(target_path):
            print(f" File [ {os.path.basename(target_path)} ] not found inside {base_folder}. Check spelling.\n")
            continue
            
        filename = os.path.basename(target_path)
        print(f" Analyzing joint rim groove of {filename} for excessive paint...")
        
        try:
            result_visual, is_defective, data = inspect_can(target_path)
            status_str = "REJECTED" if is_defective else "PASSED"
            
            print("-" * 65)
            print(f"{'Processed File':<25} | {'Inspection Status':<15} | {'Paint Pools':<15}")
            print("-" * 65)
            print(f"{filename:<25} | {status_str:<15} | {data['excessive_paint_pools']:<15}")
            print("-" * 65)
            
            output_file_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(output_file_path, result_visual)
            print(f"I/O Log: Saved marked image to [ {output_file_path} ]")
            
            print("Opening visualization canvas... (Close the image window to proceed)")
            os.system(f'start /wait "" "{output_file_path}"')
                
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f" Process Error: {e}\n")
