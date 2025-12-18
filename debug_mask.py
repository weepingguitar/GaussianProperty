import cv2
import numpy as np
import os

def analyze_mask(image_path):
    print(f"Analyzing mask: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image")
        return

    # Convert to grayscale/binary mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Calculate Area
    area = np.sum(mask > 0)
    print(f"Area: {area}")
    
    # Calculate Contours for Shape Analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found")
        return

    cnt = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    
    print(f"Contour Area: {contour_area}")
    print(f"Hull Area: {hull_area}")
    
    if hull_area > 0:
        solidity = contour_area / hull_area
        print(f"Solidity: {solidity:.4f}")
    else:
        print("Solidity: N/A (Hull Area is 0)")

    # Bounding Box
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h if h > 0 else 0
    print(f"Bounding Box: {w}x{h}")
    print(f"Aspect Ratio: {aspect_ratio:.4f}")

if __name__ == "__main__":
    mask_path = r"gp_cases_dirs\golf_ball\vis_seg\001\part\mask_0.png"
    analyze_mask(mask_path)