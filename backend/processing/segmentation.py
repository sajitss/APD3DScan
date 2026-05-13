import cv2
import numpy as np
import os

def create_roi_mask(image, points, margin_percent=0.2):
    """
    Create a mask based on the convex hull of marker points.
    points: list of (4, 2) arrays
    """
    if not points:
        return None

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Flatten all points into one array
    all_pts = []
    for p in points:
        for pt in p:
            all_pts.append(pt)
    
    all_pts = np.array(all_pts, dtype=np.int32)
    
    # Calculate convex hull
    hull = cv2.convexHull(all_pts)
    
    # Expand hull slightly
    # Calculate centroid
    M = cv2.moments(hull)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Scale hull points away from centroid
        expanded_hull = []
        for pt in hull:
            px, py = pt[0]
            dx = px - cX
            dy = py - cY
            expanded_hull.append([[int(cX + dx * (1 + margin_percent)), int(cY + dy * (1 + margin_percent))]])
        hull = np.array(expanded_hull, dtype=np.int32)

    cv2.fillPoly(mask, [hull], 255)
    return mask

def apply_mask(image, mask):
    if mask is None:
        return image
    return cv2.bitwise_and(image, image, mask=mask)

def process_segmentation(images_dir, marker_results, output_dir):
    """
    Create masked images based on detected markers.
    marker_results: list of dicts with 'image' and 'markers' (from report['per_image_results'])
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for res in marker_results:
        img_name = res["image"]
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        marker_points = [np.array(m["corners"]) for m in res["markers"]]
        
        if marker_points:
            mask = create_roi_mask(img, marker_points)
            masked_img = apply_mask(img, mask)
            cv2.imwrite(os.path.join(output_dir, f"masked_{img_name}"), masked_img)
        else:
            # If no markers, we might want to save the original or skip
            # For now, let's skip or save a black image to indicate no ROI
            pass
