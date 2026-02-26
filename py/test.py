#test and save de output
import cv2
import numpy as np
import os

def analyze_pea_leaf_and_save(image_path, output_dir="visual_checks"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load image (OpenCV loads in BGR format, which is perfect for saving later)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load {image_path}")
        return
        
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 2. Isolate the TRUE leaflet
    lower_plant = np.array([5, 75, 75])
    upper_plant = np.array([100, 255, 255])
    leaf_mask = cv2.inRange(img_hsv, lower_plant, upper_plant)

    # 3. Apply the Damage Threshold
    lower_green = np.array([30, 40, 20])
    upper_green = np.array([90, 255, 255])
    healthy_mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # The diseased area is the leaf mask minus the healthy mask
    diseased_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(healthy_mask))

    # 4. Calculate Area
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    healthy_pixels = cv2.countNonZero(healthy_mask)
    diseased_pixels = cv2.countNonZero(diseased_mask)
    
    disease_percentage = 0
    if total_leaf_pixels > 0:
        disease_percentage = (diseased_pixels / total_leaf_pixels) * 100

    print(f"--- Results for {image_path} ---")
    print(f"Severity: {disease_percentage:.2f}%\n")

    # 5. Create and Save the Visual Sanity Check
    # Extract tissues using the BGR image so saved colors are correct
    healthy_extracted = cv2.bitwise_and(img_bgr, img_bgr, mask=healthy_mask)
    diseased_extracted = cv2.bitwise_and(img_bgr, img_bgr, mask=diseased_mask)

    # Convert the single-channel gray mask to a 3-channel BGR image 
    # so we can stitch it horizontally with the other color images
    leaf_mask_bgr = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)

    # Stitch the four images together side-by-side
    
    comparison_img = cv2.hconcat([img_bgr, leaf_mask_bgr, healthy_extracted, diseased_extracted])

    # Add text displaying the severity percentage in the top left corner
    cv2.putText(comparison_img, f"Severity: {disease_percentage:.2f}%", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    # Save to the output folder
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"check_{base_name}")
    cv2.imwrite(save_path, comparison_img)
    print(f"Saved visual check to: {save_path}")

# --- RUN THE TEST ---
test_image = "InocII_sAUDPC_all/R1/6_R1_5dpi.jpg" 
analyze_pea_leaf_and_save(test_image)

