import cv2
import numpy as np
import os

def test_advanced_segmentation(image_path, output_dir="visual_checks"):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load image and convert to HSV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load {image_path}")
        return
        
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 2. Isolate the TRUE leaf (Captures everything from brown to green)
    lower_plant = np.array([5, 75, 75])
    upper_plant = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(img_hsv, lower_plant, upper_plant)

    # 3. Isolate Chlorosis (Yellow hues: ~30 to 35)
    lower_chlorosis = np.array([30, 40, 20])
    upper_chlorosis = np.array([35, 255, 255])
    chlorosis_mask = cv2.inRange(img_hsv, lower_chlorosis, upper_chlorosis)

    # 4. Isolate Necrosis (Brown/Dark hues: ~5 to 29)
    lower_necrosis = np.array([5, 40, 20])
    upper_necrosis = np.array([29, 255, 255])
    necrosis_mask = cv2.inRange(img_hsv, lower_necrosis, upper_necrosis)
    
    # 5. Isolate Healthy (Green hues: ~36 to 90)
    lower_healthy = np.array([36, 40, 20])
    upper_healthy = np.array([90, 255, 255])
    healthy_mask = cv2.inRange(img_hsv, lower_healthy, upper_healthy)

    # Ensure masks only apply strictly within the boundaries of the isolated leaf
    chlorosis_mask = cv2.bitwise_and(chlorosis_mask, leaf_mask)
    necrosis_mask = cv2.bitwise_and(necrosis_mask, leaf_mask)
    healthy_mask = cv2.bitwise_and(healthy_mask, leaf_mask)

    # 6. Calculate Areas and Percentages
    total_pixels = cv2.countNonZero(leaf_mask)
    chlorosis_pixels = cv2.countNonZero(chlorosis_mask)
    necrosis_pixels = cv2.countNonZero(necrosis_mask)
    healthy_pixels = cv2.countNonZero(healthy_mask)
    
    perc_chlorosis = (chlorosis_pixels / total_pixels * 100) if total_pixels > 0 else 0
    perc_necrosis = (necrosis_pixels / total_pixels * 100) if total_pixels > 0 else 0
    perc_healthy = (healthy_pixels / total_pixels * 100) if total_pixels > 0 else 0

    print(f"--- Results for {image_path} ---")
    print(f"Total Leaf Pixels: {total_pixels}")
    print(f"Healthy:   {perc_healthy:.2f}%")
    print(f"Chlorosis: {perc_chlorosis:.2f}%")
    print(f"Necrosis:  {perc_necrosis:.2f}%\n")

    # 7. Create the Visual Check Panels
    chlorosis_ext = cv2.bitwise_and(img_bgr, img_bgr, mask=chlorosis_mask)
    necrosis_ext = cv2.bitwise_and(img_bgr, img_bgr, mask=necrosis_mask)
    leaf_mask_bgr = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)

    # Copy images to safely add text without altering the variables
    img_titled = img_bgr.copy()
    leaf_titled = leaf_mask_bgr.copy()
    chlorosis_titled = chlorosis_ext.copy()
    necrosis_titled = necrosis_ext.copy()

    # Add the requested titles (adjust the '2' and '5' if the font is too large or thick)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (273, 41, 57) # Red text
    cv2.putText(img_titled, "raw image", (30, 80), font, 2, text_color, 5)
    cv2.putText(leaf_titled, "isolated leaf", (30, 80), font, 2, text_color, 5)
    cv2.putText(chlorosis_titled, "clorosis", (30, 80), font, 2, text_color, 5)
    cv2.putText(necrosis_titled, "necrosis", (30, 80), font, 2, text_color, 5)

    # Stitch the four images together horizontally
    comparison_img = cv2.hconcat([img_titled, leaf_titled, chlorosis_titled, necrosis_titled])

    # Save to output folder
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"chlorosis_check_{base_name}")
    cv2.imwrite(save_path, comparison_img)
    print(f"Saved visual check to: {save_path}")

# --- RUN THE TEST ---
test_image = "InocII_sAUDPC_all/R1/6_R1_8dpi.jpg" 
test_advanced_segmentation(test_image)
