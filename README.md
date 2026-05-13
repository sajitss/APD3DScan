# 3D Scan Prototype: Marker-Assisted Reconstruction

This prototype application facilitates 3D scanning of human limbs (or similar objects) using a smartphone camera and 1-inch square QR code markers. It provides a capture workflow, quality reporting, and ROI-based background suppression.

## Features
- **Mobile-First Workflow**: Capture 20 images at 2-second intervals while moving around the subject.
- **QR Code Markers**: Uses 1-inch (25.4 mm) square QR codes for scale estimation and ROI identification.
- **ROI Masking**: Automatically creates convex hull masks around detected markers to focus on the subject.
- **Quality Reporting**: Provides feedback on marker coverage, scale (mm/pixel), and capture consistency.
- **Extensible Architecture**: Clean stubs for COLMAP or other SfM/MVS integration.

## Project Structure
```
project/
  backend/
    main.py           # FastAPI server & pipeline orchestration
    processing/
      marker_detection.py # QR detection & quality report
      segmentation.py     # ROI masking logic
      reconstruction.py   # SfM stub (COLMAP placeholder)
    uploads/          # Raw images storage
    outputs/          # Debug images, masks, and reports
  frontend/
    index.html        # Main UI
    app.js            # Camera & capture logic
    styles.css        # Premium styling
```

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
cd backend
pip install -r requirements.txt
```

### 2. Networking (Laptop to Phone)
To access the app from your phone:
1. Connect both laptop and phone to the same Wi-Fi.
2. Find your laptop's local IP address:
   - **Windows**: Run `ipconfig` in CMD/PowerShell (look for IPv4 Address).
   - **Mac/Linux**: Run `ifconfig` or `ip addr`.
3. The server will run on port `8000`.

### 3. HTTPS Requirement
Mobile browsers require HTTPS for camera access (`getUserMedia`). For local development:
- Use **[mkcert](https://github.com/FiloSottile/mkcert)** to generate a local certificate.
- Or run a reverse proxy like **ngrok** to get a public HTTPS URL: `ngrok http 8000`.

### 4. Run the Backend
```bash
python main.py
```
The application will be available at `http://<your-laptop-ip>:8000`.

## Usage Guide
1. **Prepare Subject**: Place at least 5-8 QR code stickers (1-inch square) around the leg.
2. **Open App**: Access the URL on your phone's browser.
3. **Grant Permissions**: Allow camera access.
4. **Start Scan**: Hold the phone 30-50cm away and click "Start Scan".
5. **Move Slowly**: Rotate around the leg as the app captures 20 photos automatically.
6. **Upload & Process**: The app will batch upload and display the quality report once processing is complete.

## Known Limitations
- **Accuracy**: This is a prototype; sub-millimeter accuracy depends on camera calibration and marker placement.
- **Reconstruction**: Full 3D point clouds/meshes require a local installation of COLMAP, which is currently stubbed.
- **Lighting**: Requires even, bright lighting for optimal QR code detection.

## Future Enhancements
- Real-time marker tracking in the browser.
- Full COLMAP/OpenMVS integration for automated mesh generation.
- Distance-to-subject guidance using marker size.
- Neural network-based limb segmentation.
