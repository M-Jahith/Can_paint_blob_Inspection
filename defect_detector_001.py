import cv2
import numpy as np
import os
import sys

def inspect_can(image_path):
    """
    High-precision inspection script optimized to isolate the flat bottom layer 
    from the side walls, capturing both floor scratches and thick excessive paint 
    pools along the groove ring while filtering out uniform line shadows.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image: {image_path}")
    
    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. DYNAMIC FLOOR LOCALIZATION (Completely isolates the bottom layer floor)
    heavy_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    _, floor_thresh = cv2.threshold(heavy_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_floor, _ = cv2.findContours(floor_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y, safe_radius = int(w / 2), int(h / 2), int(min(h, w) / 4.3)
    
    if contours_floor:
        largest_cnt = max(contours_floor, key=cv2.contourArea)
        (x_circle, y_circle), radius = cv2.minEnclosingCircle(largest_cnt)
        center_x, center_y = int(x_circle), int(y_circle)
        
        # 95% radius retains the critical outer groove of the bottom floor while cutting off the vertical walls
        safe_radius = int(radius * 0.95)
        cv2.circle(mask, (center_x, center_y), safe_radius, 255, -1)
    else:
        cv2.circle(mask, (center_x, center_y), safe_radius, 255, -1)

    # 2. Preprocessing & Contrast Balancing
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Lowered C to 8 to allow the contrast signatures of heavy paint pools to pass through
    thresh_local = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=35, C=8  
    )
    
    # Focus strictly on the floor zone
    defect_zone = cv2.bitwise_and(thresh_local, mask)

    # Erase micro dust specks and pixel noise fragments
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_zone = cv2.morphologyEx(defect_zone, cv2.MORPH_OPEN, kernel_clean)

    defects_found = False
    analytics = {"total_defects": 0}

    contours, _ = cv2.findContours(defect_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Lowered area limit to 18px to reliably capture medium paint accumulations and deep scratches
        if 18 <= area <= 1500:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Calculate shape compactness geometry
            compactness = (4 * np.pi * area) / (perimeter ** 2)

            # SHADOW LINE SEPARATION RULE:
            # Long, paper-thin shadow segments yield compactness values near 0.
            # True excessive paint pools break away from the groove cleanly, producing higher values.
            if max(w_box, h_box) > 40 and compactness < 0.08:
                continue  # Drops normal line shadows safely

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # True paint pool anomaly or scratch verified
            defects_found = True
            analytics["total_defects"] += 1
            
            # Map visual markers
            cv2.drawContours(output_img, [cnt], -1, (255, 0, 0), 2)  # Blue outline marker
            cv2.circle(output_img, (cX, cY), 4, (0, 0, 255), -1)      # Red center dot

    # Display Status Panels
    status_text = "REJECTED" if defects_found else "PASSED"
    status_color = (0, 0, 255) if defects_found else (0, 255, 0)
    cv2.putText(output_img, f"Status: {status_text}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.3, status_color, 2)
    cv2.putText(output_img, f"Defects Found: {analytics['total_defects']}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    return output_img, defects_found, analytics


if __name__ == "__main__":
    base_folder = r"E:\W4Labs\CanInspection"
    output_dir = os.path.join(base_folder, "Inspection_Results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 65)
    print("         CAN MANUFACTURING QUALITY INSPECTION SYSTEM         ")
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
            print(f" File [ {os.path.basename(target_path)} ] not found inside {base_folder}. Check extension/spelling.\n")
            continue
            
        filename = os.path.basename(target_path)
        print(f" Analyzing bottom layer profile of {filename}...")
        
        try:
            result_visual, is_defective, data = inspect_can(target_path)
            status_str = "REJECTED" if is_defective else "PASSED"
            
            print("-" * 65)
            print(f"{'Processed File':<25} | {'Inspection Status':<15} | {'Defect Spots':<15}")
            print("-" * 65)
            print(f"{filename:<25} | {status_str:<15} | {data['total_defects']:<15}")
            print("-" * 65)
            
            output_file_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(output_file_path, result_visual)
            print(f"I/O Log: Saved marked image to [ {output_file_path} ]")
            
            print("Opening visualization canvas... (Close the image window to proceed)")
            os.system(f'start /wait "" "{output_file_path}"')
                
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f" Process Error: {e}\n")
