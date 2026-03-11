import os
from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import sys
import json
from datetime import datetime
import socket
import pickle
import time
from ultralytics import YOLO
# UR5 Robot IP configuration
ROBOT_IP = '192.168.0.1' # UR5 IP address
PORT = 3000          # Standard port for URScript communication
HOST_IP =  '0.0.0.0'


# Settings
frameWidth = 1280  
frameHeight = 720
OUTPUT_DIR = "shapes_detected"

# Create main window
cv.namedWindow("Shapes Detected", cv.WINDOW_NORMAL)
cv.resizeWindow("Shapes Detected", 1280,720)

# Global list to store detected rectangles
detected_shapes = []

# confidence threshold for shape detection
confidence_threshold = 0.5

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

#classification
def get_contours(img, img_dilate, img_contour, area_min, yolo_model): 
    """Find contours, collect relevant shapes with metadata and draw them.

    Uses cv.minAreaRect to compute center, width, height, angle and the rotated
    bounding box points, avoiding axis-aligned cv.boundingRect.
    """
    global detected_shapes
    detected_shapes = []  # Reset list for each frame
    
    contours, hierarchy = cv.findContours(img_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    results = yolo_model(img)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area> 20000:
            angle_z = 0
            # Approximate the contour to a polygon
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.01 * peri, True)
            
            # Use minimum-area rotated rect for metrics
            (center, (width, height), angle) = cv.minAreaRect(cnt)
            cx, cy = int(center[0]), int(center[1])
            print("area"+str(area))
            # Determine shape 
            if len(approx) == 4 and width > 0 and height > 0 and  area <=area_min:
                shape =    "Rectangle"
            elif area > area_min and len(approx) > 4: 
                shape = "Rectangle"
                
                boxes = results[0].obb  # OBB (Oriented Bounding Boxes)
                if boxes is not None and len(boxes) > 0:
                    print(f"DEBUG - Found {len(boxes)} detections")
                    
                    # Get all confidences, OBBs, and angles at once
                    confidences = boxes.conf.cpu().numpy()
                    obbs = boxes.xyxyxyxy.cpu().numpy()  # All 4 corner points for all boxes
                    
                    # Check if angles are available
                    if hasattr(boxes, 'xywhr') and boxes.xywhr is not None:
                        angles = boxes.xywhr.cpu().numpy()[:, 4]  # All rotation angles
                    else:
                        angles = np.zeros(len(confidences))
                    print(len(confidences))
                    
                    # Draw all detections above threshold, but append only the highest-confidence one
                    best_idx = None
                    best_conf = -1.0
                    for i in range(len(confidences)):
                        conf = confidences[i]
                        if conf <= confidence_threshold:
                            continue

                        if conf > best_conf:
                            best_conf = conf
                            best_idx = i

                        obb = obbs[i]  # 4 corner points for this detection

                        # Calculate center from OBB corners
                        cx = float(np.mean(obb[:, 0]))
                        cy = float(np.mean(obb[:, 1]))
                        
                        # Draw OBB on img_display
                        obb_points = obb.astype(int)
                        cv.polylines(img_contour, [obb_points], isClosed=True, color=(0, 255, 0), thickness=3)
                        
                        # Draw center point
                        cv.circle(img_contour, (int(cx), int(cy)), 7, (0, 0, 255), -1)
                        
                        # Display confidence
                        cv.putText(img_contour, f"Conf: {int(conf * 100)}%", 
                                (int(cx), int(cy) - 20),
                                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Display center coordinates
                        cv.putText(img_contour, f"({int(cx)},{int(cy)})", 
                                (int(cx), int(cy) + 30),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if best_idx is not None:
                        conf = confidences[best_idx]
                        obb = obbs[best_idx]
                        angle_z = float(angles[best_idx])

                        # Aus den 4 OBB-Punkten
                        obb_points = obb.astype(np.int32)

                        # minAreaRect gibt direkt den Winkel zurück
                        Center, (width,height), angle = cv.minAreaRect(obb_points)
                        if width < height:
                            angle_z = 90 - (90-angle)

                        else:
                            angle_z= (90-angle) * (-1)

                        cx = float(np.mean(obb[:, 0]))
                        cy = float(np.mean(obb[:, 1]))

                        angle_z = float(angle_z)
                        if  0<cx<950 and 0<cy :

                            detected_shapes.append({
                                "cx": cx,
                                "cy": cy,
                                "angle": angle_z,
                                "confidence": float(conf),
                                "shape": shape
                            })
                        print(f"Detected #{i+1} - Center: ({cx:.1f}, {cy:.1f}), Angle: {angle_z:.2f}, Confidence: {conf*100:.1f}%")

                        
                break
            elif area> area_min and len(approx)==4:
                shape ="bug"
                if width < height:
                    angle_z = 90 - (90-angle)

                else:
                    angle_z= (90-angle) * (-1)

                if  0<cx<950 and 0<cy:
                    detected_shapes.append({
                        "cx": cx,
                        "cy": cy,
                        "angle": float(angle_z),
                        "shape": shape,
                        "confidence": 0
                    })
                break           
            else:
                shape = "not relevant"

            if width < height:
                angle_z = 90 - (90-angle)
            else:
                angle_z= (90-angle) * (-1)

            if  0<cx<950 and 0<cy:
                detected_shapes.append({
                    "cx": cx,
                    "cy": cy,
                    "angle": float(angle_z),
                    "shape": shape,
                    "confidence": 0
                })
            
                cv.drawContours(img_contour, [cnt], -1, (255, 0, 255), 3)
                                
                # Get bounding box for text placement
                x, y, w, h = cv.boundingRect(approx)
                
                
                # Draw center point
                cv.circle(img_contour, (cx, cy), 5, (0, 0, 255), -1)
                
                # Display shape information
                cv.putText(img_contour, shape, (x, y - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.putText(img_contour, f"Center: ({cx},{cy})", (x, y + h + 20),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.putText(img_contour, f"Angle: {int(angle_z)} ", (x, y + h + 40),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def load_parameters(filename="parameters.json"):
    """Load parameters from JSON file; fall back to defaults if not found."""
    defaults = {
        "threshold1": 50,
        "threshold2": 150,
        "area_min": 1000
    }
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                threshold1 = data.get("threshold1", defaults["threshold1"])
                threshold2 = data.get("threshold2", defaults["threshold2"])
                area_min = data.get("area_min", defaults["area_min"])
                print(f"Loaded parameters from {filename}:")
                print(f"  Threshold1: {threshold1}")
                print(f"  Threshold2: {threshold2}")
                print(f"  Area Min: {area_min}")
                return threshold1, threshold2, area_min
        except Exception as e:
            print(f"Error loading {filename}: {e}. Using defaults.")
    else:
        print(f"Parameter file {filename} not found. Using defaults.")
    
    return defaults["threshold1"], defaults["threshold2"], defaults["area_min"]

def send_pose(conn, pose):
    """Send pose data to the UR5 robot via socket"""
    try:
        if pose:
            pose_array= np.array(pose.split(';'))
            for value in pose_array:
                val_s = str(value)
                conn.sendall(val_s.encode())
                print(f"Pose sent: {value}")
                time.sleep(0.1)
    except Exception as e:
        print(f"Error sending pose: {e}")

def pose_processing():
    """Extract pose from detected rectangles"""
    if detected_shapes:
        
        cx= detected_shapes[0]["cx"]
        cy= detected_shapes[0]["cy"]
        angle = detected_shapes[0]["angle"]
        shape = detected_shapes[0]["shape"]
        shape_value = 0 
        # Points come from the already-undistorted frame, use them directly
        p = np.array([float(cx), float(cy), 1.0])
        target = np.dot(H,p) # H = Transformationsmatrix
        target /= target[2] # Normalisierung       
        print("target"+str(target))
        # Normalisieren (durch z teilen)

        target_x = target[0] / 1000
        target_y = target[1] /1000
        if target_x <= 0:
            target_x_string = f"{target_x:.3f}"
        else:
            target_x_string = f"{target_x:.4f}"
        if target_y <= 0:
            target_y_string=f"{target_y:.3f}"
        else:
            target_y_string=f"{target_y:.4f}"
            
        if shape == "Rectangle":
            shape_value = 1
        elif shape == "bug":
            shape_value = 2
        else:
            shape_value = 0
        
        pose = f"{target_x_string}{target_y_string}{shape_value}{angle:.4f}"
        detected_shapes.remove(detected_shapes[0])
        print(pose)
        return pose

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

# Parse command-line argument for parameter file
param_file = sys.argv[1] if len(sys.argv) > 1 else "parameters.json"
threshold1, threshold2, area_min = load_parameters(param_file)

# load calibration_data.pkl
with open('calibration_data.pkl','rb') as f:
    data = pickle.load(f)
mtx = data['camera_matrix']
dist = data['distortion_coefficients']

# load homography matrix
with open('homography_matrix.pkl', 'rb') as f:
    H = pickle.load(f)
print("Homography matrix loaded successfully.")

#load model
yolo = YOLO("/home/aas/Documents/ur5_project/runs/obb/bag_project/yolo26_first_run6/weights/best.pt")

# Initialize Picamera2
print("Initializing Pi Camera...")
picam2 = Picamera2()

# Configure camera for HD video
config = picam2.create_preview_configuration(
    main={"size": (frameWidth, frameHeight), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("Press Ctrl+C to quit.")

try:
        # Setup socket server for UR5 communication
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST_IP, PORT))
        s.listen(1)
        print(f"Listening for UR5 connection on port {PORT}...")
        conn, address = s.accept()
        print(f"UR5 connected from {address}")
        # Set socket to non-blocking for non-intrusive request checking
        conn.setblocking(False)
    while True:
        # Capture frame from camera (RGB)
        img = picam2.capture_array()

        # Undistort the captured frame before any processing
        img = cv.undistort(img, mtx, dist, None, mtx)

        # Make a copy for contour drawing
        img_contour = img.copy()

        
        # Image processing pipeline
        img_blur = cv.GaussianBlur(img, (7, 7), 1)
        img_gray = cv.cvtColor(img_blur, cv.COLOR_RGB2GRAY)
        
        # Edge detection
        img_canny = cv.Canny(img_gray, threshold1, threshold2)
        
        # Morphological operations to clean up edges
        kernel = np.ones((5, 5))
        img_dilate = cv.dilate(img_canny, kernel, iterations=1)
        
        # Detect shapes 
        get_contours(img,img_dilate, img_contour, area_min, yolo)
        

        # Check for incoming requests from UR5 
        if conn:
            try:
                data = conn.recv(1024)
                if data:
                    request = data.decode().strip()
                    print(f"Received from UR5: {request}")
                    if "READY" in request:
                        pose = pose_processing()
                        send_pose(conn, pose)
                    else:
                        print("fehler")
            except BlockingIOError:
                # No request available; continue frame processing
                pass



        # Stack images for comparison view
        img_gray_3ch = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
        img_canny_3ch = cv.cvtColor(img_canny, cv.COLOR_GRAY2BGR)
        img_dilate_3ch = cv.cvtColor(img_dilate, cv.COLOR_GRAY2BGR)
        
        img_stack = stack_images(0.5, ([img, img_gray_3ch],
                                       [img_canny_3ch, img_dilate_3ch]))
        
        
        # Display results
        cv.imshow("Shapes Detected",img_contour)
        
        # Wait for key press (required for window updates)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping camera...")

finally:
    # Cleanup
    picam2.stop()
    cv.destroyAllWindows()
    if conn:
        conn.close()
    if s:
        s.close()
    print("Camera stopped and socket closed.")
