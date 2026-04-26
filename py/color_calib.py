import cv2
import numpy as np
import os
import glob

def build_color_model(checker_path):
    """Detects the color chart and builds a transformation model."""
    img_bgr = cv2.imread(checker_path)
    if img_bgr is None:
        print(f"Error: Could not read {checker_path}")
        return None
        
    print(f"  -> Extracting color profile from {os.path.basename(checker_path)}...")
    
    detector = cv2.mcc.CCheckerDetector_create()
    success = detector.process(img_bgr, cv2.mcc.MCC24, 1) 
    
    if not success:
        print(f"  [!] Failed to auto-detect color patches in {os.path.basename(checker_path)}.")
        return None

    checker = detector.getBestColorChecker()
    charts_rgb = checker.getChartsRGB()

    # THE FIX: Extract the detected colors, reshape the array, and scale to 0.0 - 1.0
    src = charts_rgb[:, 1].copy().reshape(24, 1, 3)
    src = src / 255.0

    # Build the model using the reshaped 'src' and the Macbeth standard target
    ccm = cv2.ccm_ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
    ccm.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
    ccm.setCCM_TYPE(cv2.ccm.CCM_3x3)
    ccm.run()
    
    return ccm

def apply_color_correction(img_bgr, ccm):
    """Applies the model ONLY to the isolated leaf, keeping background white."""
    
    # 1. Create the mask to isolate the true leaf from the white/agar background
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_plant = np.array([5, 40, 20])
    upper_plant = np.array([100, 255, 255])
    leaf_mask = cv2.inRange(img_hsv, lower_plant, upper_plant)

    # 2. Apply the mathematical color transformation to the whole image array
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float64) / 255.0
    
    calibrated_float = ccm.infer(img_float)
    
    calibrated_float = calibrated_float * 255.0
    calibrated_rgb = np.clip(calibrated_float, 0, 255).astype(np.uint8)
    calibrated_bgr = cv2.cvtColor(calibrated_rgb, cv2.COLOR_RGB2BGR)

    # 3. Extract ONLY the calibrated leaf pixels
    calibrated_leaf = cv2.bitwise_and(calibrated_bgr, calibrated_bgr, mask=leaf_mask)

    # 4. Extract ONLY the original background pixels
    # (bitwise_not flips the mask, so the leaf is black and the background is white)
    bg_mask = cv2.bitwise_not(leaf_mask)
    original_bg = cv2.bitwise_and(img_bgr, img_bgr, mask=bg_mask)

    # 5. Merge the calibrated leaf and the original background together
    final_img = cv2.add(calibrated_leaf, original_bg)

    return final_img

def batch_calibrate(input_dir, output_dir="InocIII_sAUDPC_all/calibrated_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Find all ColorChecker images (Catching both upper and lowercase extensions)
    checker_pattern = os.path.join(input_dir, "0Datacolor_*.[jJ][pP][gG]")
    checker_files = glob.glob(checker_pattern)

    if not checker_files:
        print("No ColorChecker images found. Check your directory path.")
        return

    # 2. Loop through each calibrator 
    for checker_path in checker_files:
        basename = os.path.basename(checker_path)
        
        # Safely extract the unique identifier without the extension
        name_without_ext = os.path.splitext(basename)[0]
        identifier = name_without_ext.replace("0Datacolor_", "")
        
        print(f"\n=== Processing Group: {identifier} ===")

        # Build the mathematical model for this specific replicate/day
        color_model = build_color_model(checker_path)
        if color_model is None:
            continue

        # 3. Find all target leaf images that share this identifier
        target_pattern = os.path.join(input_dir, f"*_{identifier}.[jJ][pP][gG]")
        target_files = glob.glob(target_pattern)
        
        if not target_files:
            print(f"  [!] No target images found matching: {target_pattern}")
            continue

        # 4. Apply correction to those specific target files and save
        for target_path in target_files:
            target_basename = os.path.basename(target_path)
            
            # Skip the calibrator image itself
            if target_basename.startswith("0Datacolor"):
                continue

            print(f"  Calibrating: {target_basename}")
            img_target = cv2.imread(target_path)
            
            if img_target is not None:
                calibrated_img = apply_color_correction(img_target, color_model)
                save_path = os.path.join(output_dir, target_basename)
                cv2.imwrite(save_path, calibrated_img)

    print("\n=== Calibration Complete ===")

# --- RUN THE SCRIPT ---
input_directory = "InocIII_sAUDPC_all/R1/"
input_directory = "InocIII_sAUDPC_all/R2/" 
input_directory = "InocIII_sAUDPC_all/R3/" 
batch_calibrate(input_directory)
