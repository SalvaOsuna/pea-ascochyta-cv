#Threshold calibrator
import cv2
import numpy as np

def empty(a):
    pass

def launch_calibrator(image_path):
    # Load the image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load {image_path}")
        return
        
    # Resize image so the window actually fits on your laptop screen
    # (Assuming original photos are large high-res images)
    scale_percent = 30 # Adjust this if the window is still too big or too small
    width = int(img_bgr.shape[1] * scale_percent / 100)
    height = int(img_bgr.shape[0] * scale_percent / 100)
    img_bgr = cv2.resize(img_bgr, (width, height))
    
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Create a window for the trackbars
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbars", 500, 300)

    # OpenCV Hue range is 0-179. Saturation and Value are 0-255.
    cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

    # Initialize trackbars with our starting guess for "Healthy Green"
    cv2.setTrackbarPos("Hue Min", "Trackbars", 35)
    cv2.setTrackbarPos("Hue Max", "Trackbars", 90)
    cv2.setTrackbarPos("Sat Min", "Trackbars", 40)
    cv2.setTrackbarPos("Val Max", "Trackbars", 255)
    cv2.setTrackbarPos("Sat Max", "Trackbars", 255)

    print("--- Calibrator Running ---")
    print("Adjust the trackbars to isolate the tissue.")
    print("Press the 'q' key on your keyboard to quit and print the final values.")

    while True:
        # Read current positions of all trackbars
        h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
        h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
        s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
        s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
        v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
        v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

        # Create arrays for the threshold ranges
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Apply the mask
        mask = cv2.inRange(img_hsv, lower, upper)
        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

        # Show the images side-by-side (Original, Mask, Extracted Result)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = cv2.hconcat([img_bgr, mask_bgr, result])
        
        cv2.imshow("Original | Mask | Extracted Result (Press 'q' to quit)", stacked)

        # Wait for the 'q' key to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n--- Final Threshold Values ---")
            print(f"Lower Bound: [{h_min}, {s_min}, {v_min}]")
            print(f"Upper Bound: [{h_max}, {s_max}, {v_max}]")
            break

    cv2.destroyAllWindows()

# --- Run it on your 1dpi image first, then an 8dpi image ---
test_image = "InocII_sAUDPC_all/R1/1_R1_1dpi.jpg"
launch_calibrator(test_image)
