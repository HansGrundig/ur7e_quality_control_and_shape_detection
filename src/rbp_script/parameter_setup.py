from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import json
import os
from datetime import datetime
import pickle

# Settings
frameWidth = 1280  # HD resolution
frameHeight = 720
shapes_detected = "Live Shape Detection"


def empty(a):
    pass

# Create trackbars for threshold adjustment
cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 640, 240)
cv.createTrackbar("Threshold1", "Parameters", 157, 255, empty)
cv.createTrackbar("Threshold2", "Parameters", 134, 255, empty)
cv.createTrackbar("Area Min", "Parameters", 6995, 40000, empty)

# Create main window
cv.namedWindow(shapes_detected, cv.WINDOW_NORMAL)
cv.resizeWindow(shapes_detected, 800, 400)

def save_parameters(threshold1, threshold2, area_min, filename="parameters.json"):
    """Save current parameter values to JSON file"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "threshold1": threshold1,
        "threshold2": threshold2,
        "area_min": area_min
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Parameters saved: {filename}")
    print(f"  Threshold1: {threshold1}")
    print(f"  Threshold2: {threshold2}")
    print(f"  Area Min: {area_min}")

def detect_shape(approx):
    """Identify the shape based on the number of vertices"""
    vertices = len(approx)
    shape = "Unknown"
    
    if vertices == 4:
        # Check if it's a square or rectangle
        (x, y, w, h) = cv.boundingRect(approx)
        aspect_ratio = float(w) / h
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    else:
        shape = "not relevant"
    return shape, vertices

def get_contours(img, img_contour):
    """Find and draw contours with shape detection"""
    
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        area_min = cv.getTrackbarPos("Area Min", "Parameters")
        
        if 4*37000 >area> area_min :
            
            # Approximate the contour to a polygon
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Get shape information
            shape, vertices = detect_shape(approx)
            
            # Get bounding box
            x, y, w, h = cv.boundingRect(approx)
            
            # Calculate center point
            cx = x + w // 2
            cy = y + h // 2

            if cx <950:

                # Draw contour
                cv.drawContours(img_contour, [cnt], -1, (255, 0, 255), 3)
                
                
                # Draw center point
                cv.circle(img_contour, (cx, cy), 5, (0, 0, 255), -1)
                
                # Get rotation angle
                if len(approx) == 4:
                    (center, (width, height), angle) = cv.minAreaRect(cnt)

                else:
                    angle = 0
                
                # Display shape information
                cv.putText(img_contour, shape, (x, y - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.putText(img_contour, f"Center: ({cx},{cy})", (x, y + h + 60),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.putText(img_contour, f"Angle: {int(angle)}", (x, y + h + 80),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.putText(img_contour, f"Area: {int(area)}", (x, y + h + 40),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def stack_images(scale, img_array):
    """Stack images in a grid for display"""
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv.resize(img_array[x][y],
                                                (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
        
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv.resize(img_array[x],
                                        (img_array[0].shape[1], img_array[0].shape[0]),
                                        None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    
    return ver

# Initialize Picamera2
print("Initializing Pi Camera...")
picam2 = Picamera2()

# Load camera calibration parameters
# load calibration_data.pkl
with open('calibration_data.pkl','rb') as f:
    data = pickle.load(f)
mtx = data['camera_matrix']
dist = data['distortion_coefficients']
# Configure camera for HD video
config = picam2.create_preview_configuration(
    main={"size": (frameWidth, frameHeight), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()



print("Press 'q' to quit.")

try:
    while True:
        # Capture frame from camera
        img = picam2.capture_array()

        # Undistort the captured frame before any processing
        img = cv.undistort(img, mtx, dist, None, mtx)
        
        # Make a copy for contour drawing
        img_contour = img.copy()


        
        # Image processing pipeline
        img_blur = cv.GaussianBlur(img, (7, 7), 1)
        img_gray = cv.cvtColor(img_blur, cv.COLOR_RGB2GRAY)
        
        # Get threshold values from trackbars
        threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")
        
        # Edge detection
        img_canny = cv.Canny(img_gray, threshold1, threshold2)
        
        # Morphological operations to clean up edges
        kernel = np.ones((5, 5))
        img_dilate = cv.dilate(img_canny, kernel, iterations=1)
        
        # Detect shapes
        get_contours(img_dilate, img_contour)
        
        # Stack images for comparison view
        img_gray_3ch = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
        img_canny_3ch = cv.cvtColor(img_canny, cv.COLOR_GRAY2BGR)
        img_dilate_3ch = cv.cvtColor(img_dilate, cv.COLOR_GRAY2BGR)
        
        img_stack = stack_images(0.5, ([img, img_gray_3ch],
                                       [img_canny_3ch, img_dilate_3ch]))
        
        
        # Display results
        cv.imshow(shapes_detected, img_contour)
        #cv.imshow("Processing Steps", img_stack)
        
        # Handle key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current parameters to JSON
            threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
            threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")
            area_min = cv.getTrackbarPos("Area Min", "Parameters")
            save_parameters(threshold1, threshold2, area_min)
            

except KeyboardInterrupt:
    print("\nStopping camera...")

finally:
    # Cleanup
    picam2.stop()
    cv.destroyAllWindows()
    print("Camera stopped and windows closed.")
