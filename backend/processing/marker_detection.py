import cv2
import numpy as np
import os
import json

class MarkerDetector:
    def __init__(self, marker_size_mm=25.4):
        self.marker_size_mm = marker_size_mm
        self.qr_detector = cv2.QRCodeDetector()

    def detect_markers(self, image_path):
        """
        Detect QR codes in the image with preprocessing for robustness.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None, [], []

        # 1. Try on original image
        retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(img)
        
        # 2. If no markers found, try preprocessing
        if not retval or points is None:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Try again on enhanced grayscale image
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(enhanced)
            
            # 3. If still no markers, try a slightly blurred version to reduce noise
            if not retval or points is None:
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(blurred)

        if not retval or points is None:
            return img, [], []

        return img, points, decoded_info

    def get_pixel_to_mm_scale(self, points):
        """
        Estimate scale (mm per pixel) based on detected QR code corners.
        Assuming square markers.
        """
        scales = []
        for marker_points in points:
            # marker_points is 4x2: [top-left, top-right, bottom-right, bottom-left]
            # Average side length in pixels
            side1 = np.linalg.norm(marker_points[0] - marker_points[1])
            side2 = np.linalg.norm(marker_points[1] - marker_points[2])
            side3 = np.linalg.norm(marker_points[2] - marker_points[3])
            side4 = np.linalg.norm(marker_points[3] - marker_points[0])
            avg_pixel_size = (side1 + side2 + side3 + side4) / 4.0
            if avg_pixel_size > 0:
                scales.append(self.marker_size_mm / avg_pixel_size)
        
        if not scales:
            return None
        return np.mean(scales)

    def draw_detections(self, img, points, decoded_info):
        """Draw detected QR codes on image for debugging."""
        debug_img = img.copy()
        for i, marker_points in enumerate(points):
            # marker_points is (4, 2)
            pts = marker_points.astype(int).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
            # Label
            label = decoded_info[i] if decoded_info[i] else f"ID:{i}"
            cv2.putText(debug_img, label, tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return debug_img

def process_scan_markers(scan_id, images_dir, output_dir, marker_size_mm=25.4):
    detector = MarkerDetector(marker_size_mm)
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    report = {
        "scan_id": scan_id,
        "images_received": len(image_files),
        "unique_markers_detected": 0,
        "images_with_markers": 0,
        "images_without_markers": 0,
        "average_marker_pixel_size": 0,
        "estimated_mm_per_pixel": 0,
        "coverage_score": "poor",
        "warnings": [],
        "per_image_results": []
    }

    all_decoded_info = set()
    total_scales = []
    total_pixel_sizes = []

    os.makedirs(output_dir, exist_ok=True)

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        img, points, decoded_info = detector.detect_markers(img_path)
        
        res = {
            "image": img_name,
            "marker_count": len(points),
            "markers": []
        }

        if len(points) > 0:
            report["images_with_markers"] += 1
            scale = detector.get_pixel_to_mm_scale(points)
            if scale:
                total_scales.append(scale)
            
            for i, p in enumerate(points):
                all_decoded_info.add(decoded_info[i])
                side_len = np.linalg.norm(p[0] - p[1])
                total_pixel_sizes.append(side_len)
                res["markers"].append({
                    "id": decoded_info[i],
                    "corners": p.tolist()
                })
            
            # Save debug image
            debug_img = detector.draw_detections(img, points, decoded_info)
            cv2.imwrite(os.path.join(output_dir, f"debug_{img_name}"), debug_img)
        else:
            report["images_without_markers"] += 1

        report["per_image_results"].append(res)

    report["unique_markers_detected"] = len(all_decoded_info)
    if total_pixel_sizes:
        report["average_marker_pixel_size"] = float(np.mean(total_pixel_sizes))
    if total_scales:
        report["estimated_mm_per_pixel"] = float(np.mean(total_scales))

    # Coverage score logic
    if report["images_with_markers"] >= 15 and report["unique_markers_detected"] >= 5:
        report["coverage_score"] = "excellent"
    elif report["images_with_markers"] >= 10 and report["unique_markers_detected"] >= 3:
        report["coverage_score"] = "good"
    elif report["images_with_markers"] > 0:
        report["coverage_score"] = "fair"

    if report["images_without_markers"] > 0:
        report["warnings"].append(f"{report['images_without_markers']} images had no detectable markers")
    
    if report["unique_markers_detected"] < 5:
        report["warnings"].append(f"Only {report['unique_markers_detected']} unique markers detected. 5+ recommended for stability.")

    return report
