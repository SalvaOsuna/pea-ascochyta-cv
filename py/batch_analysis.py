import cv2
import numpy as np
import os
import glob
import csv

def analyze_calibrated_images(input_dir, output_csv="pea_phenotype_results.csv", output_img_dir="batch_visual_checks"):
    # Create the output directory for the images
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        
    # 1. Prepare the CSV Headers
    headers = [
        "Filename", "Genotype", "Replicate", "DPI", 
        "Total_Pixels", "Healthy_Perc", "Chlorosis_Perc", "Necrosis_Perc",
        "Morph_Area", "Morph_Perimeter", "Major_Axis", "Minor_Axis", 
        "Aspect_Ratio", "Circularity", "Solidity"
    ]
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        # 2. Find all images in the directory
        pattern = os.path.join(input_dir, "*.[jJ][pP][gG]")
        image_files = glob.glob(pattern)

        if not image_files:
            print(f"No images found in {input_dir}.")
            return

        print(f"Found {len(image_files)} images. Starting batch analysis...\n")

        # 3. Process each image
        for img_path in image_files:
            basename = os.path.basename(img_path)
            
            # Parse the experimental design from the filename (e.g., "1_R1_1dpi")
            name_without_ext = os.path.splitext(basename)[0]
            try:
                parts = name_without_ext.split('_')
                genotype = parts[0]
                replicate = parts[1]
                dpi = parts[2].replace('dpi', '')
            except IndexError:
                genotype, replicate, dpi = "NA", "NA", "NA"

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
                
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # --- TISSUE SEGMENTATION ---
            # 1. Isolate the TRUE leaf (From Attached 1: Hue 0-179, Sat 10-255, Val 50-200)
            lower_plant = np.array([0, 9, 50])
            upper_plant = np.array([179, 255, 200])
            leaf_mask = cv2.inRange(img_hsv, lower_plant, upper_plant)

            # 2. Isolate Healthy (From Attached 2: Hue 35-60)
            lower_healthy = np.array([46, 10, 50])
            upper_healthy = np.array([179, 255, 200])
            healthy_mask = cv2.inRange(img_hsv, lower_healthy, upper_healthy)

            # 3. Isolate Chlorosis (Top half of Attached 3: Hue 30-34)
            lower_chlorosis = np.array([30, 10, 50])
            upper_chlorosis = np.array([45, 255, 200])
            chlorosis_mask = cv2.inRange(img_hsv, lower_chlorosis, upper_chlorosis)

            # 4. Isolate Necrosis (Bottom half of Attached 3: Hue 0-29)
            lower_necrosis = np.array([1, 10, 50])
            upper_necrosis = np.array([29, 255, 200])
            necrosis_mask = cv2.inRange(img_hsv, lower_necrosis, upper_necrosis)

            # Ensure masks only apply strictly within the boundaries of the isolated leaf
            chlorosis_mask = cv2.bitwise_and(chlorosis_mask, leaf_mask)
            necrosis_mask = cv2.bitwise_and(necrosis_mask, leaf_mask)
            healthy_mask = cv2.bitwise_and(healthy_mask, leaf_mask)

            total_pixels = cv2.countNonZero(leaf_mask)
            if total_pixels == 0:
                # If no leaf is found, write zeros and skip to the next image
                writer.writerow([basename, genotype, replicate, dpi] + [0]*11)
                continue

            healthy_perc = (cv2.countNonZero(healthy_mask) / total_pixels) * 100
            chlorosis_perc = (cv2.countNonZero(chlorosis_mask) / total_pixels) * 100
            necrosis_perc = (cv2.countNonZero(necrosis_mask) / total_pixels) * 100

            # --- GEOMETRIC MORPHOLOGY ---
            # Find the outer boundary of the leaf to calculate geometry
            contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                writer.writerow([basename, genotype, replicate, dpi, total_pixels, healthy_perc, chlorosis_perc, necrosis_perc] + [0]*7)
                continue

            # Grab the largest contour (ignores any tiny microscopic dust specs left on the agar)
            main_contour = max(contours, key=cv2.contourArea)
            
            morph_area = cv2.contourArea(main_contour)
            morph_perimeter = cv2.arcLength(main_contour, True)

            # Calculate axes and aspect ratio (requires at least 5 points to fit an ellipse)
            major_axis, minor_axis, aspect_ratio = 0, 0, 0
            if len(main_contour) >= 5:
                (x, y), (minor_axis, major_axis), angle = cv2.fitEllipse(main_contour)
                if minor_axis > 0:
                    aspect_ratio = major_axis / minor_axis

            # Calculate Circularity
            circularity = 0
            if morph_perimeter > 0:
                circularity = 4 * np.pi * (morph_area / (morph_perimeter * morph_perimeter))

            # Calculate Solidity (Convex Hull ratio)
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = 0
            if hull_area > 0:
                solidity = morph_area / hull_area
            
            # --- VISUAL CHECK GENERATION ---
            # Extract the colored pixels using the masks
            chlorosis_ext = cv2.bitwise_and(img_bgr, img_bgr, mask=chlorosis_mask)
            necrosis_ext = cv2.bitwise_and(img_bgr, img_bgr, mask=necrosis_mask)
            
            # Convert the single-channel leaf mask to 3-channel BGR for concatenation
            leaf_mask_bgr = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)

            # Copy images to safely add text
            img_titled = img_bgr.copy()
            leaf_titled = leaf_mask_bgr.copy()
            chlorosis_titled = chlorosis_ext.copy()
            necrosis_titled = necrosis_ext.copy()

            # Add the titles
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (273, 41, 57) # Blue text
            cv2.putText(img_titled, "raw image", (30, 80), font, 2, text_color, 5)
            cv2.putText(leaf_titled, "isolated leaf", (30, 80), font, 2, text_color, 5)
            cv2.putText(chlorosis_titled, "chlorosis", (30, 80), font, 2, text_color, 5)
            cv2.putText(necrosis_titled, "necrosis", (30, 80), font, 2, text_color, 5)

            # Stitch them together
            comparison_img = cv2.hconcat([img_titled, leaf_titled, chlorosis_titled, necrosis_titled])

            # Save to the visual checks folder
            save_path = os.path.join(output_img_dir, f"check_{basename}")
            cv2.imwrite(save_path, comparison_img)
            # 4. Write all data to the CSV
            writer.writerow([
                basename, genotype, replicate, dpi,
                total_pixels, 
                round(healthy_perc, 4), round(chlorosis_perc, 4), round(necrosis_perc, 4),
                round(morph_area, 2), round(morph_perimeter, 2), 
                round(major_axis, 2), round(minor_axis, 2), 
                round(aspect_ratio, 4), round(circularity, 4), round(solidity, 4)
            ])
            print(f"Processed: {basename}")

    print(f"\n=== Analysis Complete! Results saved to {output_csv} ===")

# --- RUN THE BATCH ANALYSIS ---
input_directory = "calibrated_images" 
analyze_calibrated_images(input_directory)
