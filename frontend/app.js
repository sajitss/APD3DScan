/**
 * 3D Scan Prototype - Frontend Logic
 */

const CONFIG = {
    CAPTURE_COUNT: 20,
    CAPTURE_INTERVAL: 2000, // ms
    API_UPLOAD_URL: '/api/upload-scan',
    API_STATUS_URL: (id) => `/api/scan-status/${id}`
};

const state = {
    stream: null,
    captures: [],
    isScanning: false,
    wakeLock: null
};

// DOM Elements
const video = document.getElementById('camera-preview');
const canvas = document.getElementById('capture-canvas');
const startBtn = document.getElementById('start-btn');
const photoCountEl = document.getElementById('photo-count');
const progressFill = document.getElementById('progress-fill');
const overlay = document.getElementById('overlay');
const countdownNum = document.getElementById('countdown-number');
const flashIndicator = document.getElementById('capture-indicator');
const statusSection = document.getElementById('status-section');
const cameraSection = document.getElementById('camera-section');
const statusTitle = document.getElementById('status-title');
const statusMsg = document.getElementById('status-msg');
const qualityReport = document.getElementById('quality-report');
const resetBtn = document.getElementById('reset-btn');
const previewGrid = document.getElementById('preview-grid');
const gallerySection = document.getElementById('gallery-section');

/**
 * Initialize Camera
 */
async function initCamera() {
    try {
        const constraints = {
            video: {
                facingMode: 'environment',
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };
        state.stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = state.stream;
        console.log("Camera initialized");
    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Camera access denied or not available. Please ensure you are using HTTPS and have granted permissions.");
    }
}

/**
 * Wake Lock to prevent screen sleep
 */
async function requestWakeLock() {
    if ('wakeLock' in navigator) {
        try {
            state.wakeLock = await navigator.wakeLock.request('screen');
        } catch (err) {
            console.warn("Wake Lock failed:", err);
        }
    }
}

/**
 * Capture single frame
 */
function captureFrame() {
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    return canvas.toDataURL('image/jpeg', 0.8);
}

/**
 * Main Scan Loop
 */
async function startScan() {
    if (state.isScanning) return;
    
    state.isScanning = true;
    state.captures = [];
    startBtn.disabled = true;
    overlay.classList.remove('hidden');
    requestWakeLock();

    for (let i = 1; i <= CONFIG.CAPTURE_COUNT; i++) {
        // Countdown
        countdownNum.innerText = i;
        
        // Wait for interval
        await new Promise(resolve => setTimeout(resolve, CONFIG.CAPTURE_INTERVAL));
        
        // Visual Flash
        flashIndicator.classList.add('flash');
        setTimeout(() => flashIndicator.classList.remove('flash'), 200);
        
        // Capture
        const imageData = captureFrame();
        state.captures.push({
            name: `scan_${i.toString().padStart(3, '0')}.jpg`,
            data: imageData
        });
        
        // Update UI
        photoCountEl.innerText = i;
        progressFill.style.width = `${(i / CONFIG.CAPTURE_COUNT) * 100}%`;
        
        console.log(`Captured ${i}/${CONFIG.CAPTURE_COUNT}`);
    }

    state.isScanning = false;
    overlay.classList.add('hidden');
    finishCapture();
}

/**
 * Prepare for upload
 */
function finishCapture() {
    cameraSection.classList.add('hidden');
    statusSection.classList.remove('hidden');
    
    // Show previews
    gallerySection.classList.remove('hidden');
    previewGrid.innerHTML = '';
    state.captures.forEach(cap => {
        const img = document.createElement('img');
        img.src = cap.data;
        previewGrid.appendChild(img);
    });

    uploadSequence();
}

/**
 * Batch Upload
 */
async function uploadSequence() {
    statusTitle.innerText = "Uploading...";
    statusMsg.innerText = `Preparing ${state.captures.length} images for transmission`;

    const formData = new FormData();
    for (const cap of state.captures) {
        const blob = await (await fetch(cap.data)).blob();
        formData.append('files', blob, cap.name);
    }

    try {
        const response = await fetch(CONFIG.API_UPLOAD_URL, {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        if (result.scan_id) {
            pollStatus(result.scan_id);
        } else {
            throw new Error("No scan ID returned");
        }
    } catch (err) {
        console.error("Upload failed:", err);
        statusTitle.innerText = "Upload Failed";
        statusMsg.innerText = "Please check your connection and try again.";
        resetBtn.classList.remove('hidden');
    }
}

/**
 * Poll Processing Status
 */
async function pollStatus(scanId) {
    statusTitle.innerText = "Processing...";
    statusMsg.innerText = "Detecting markers and preparing ROI masks";

    const interval = setInterval(async () => {
        try {
            const response = await fetch(CONFIG.API_STATUS_URL(scanId));
            const data = await response.json();
            
            if (data.status === 'reconstruction_completed') {
                clearInterval(interval);
                showResult(data);
            } else if (data.status.startsWith('failed')) {
                clearInterval(interval);
                statusTitle.innerText = "Processing Failed";
                statusMsg.innerText = data.status;
                resetBtn.classList.remove('hidden');
            } else {
                statusMsg.innerText = `Status: ${data.status.replace(/_/g, ' ')}`;
            }
        } catch (err) {
            console.warn("Poll error:", err);
        }
    }, 2000);
}

/**
 * Display Quality Report
 */
function showResult(data) {
    statusTitle.innerText = "Scan Complete";
    statusMsg.innerText = "Processing sequence finalized. View report below.";
    
    const report = data.quality_report;
    if (report) {
        qualityReport.classList.remove('hidden');
        qualityReport.innerHTML = `
            <div class="report-item">
                <span class="report-label">Images Processed</span>
                <span class="report-value">${report.images_received}</span>
            </div>
            <div class="report-item">
                <span class="report-label">Unique Markers</span>
                <span class="report-value">${report.unique_markers_detected}</span>
            </div>
            <div class="report-item">
                <span class="report-label">MM per Pixel</span>
                <span class="report-value">${report.estimated_mm_per_pixel ? report.estimated_mm_per_pixel.toFixed(3) : 'N/A'}</span>
            </div>
            <div class="report-item">
                <span class="report-label">Coverage Score</span>
                <span class="report-value score-${report.coverage_score}">${report.coverage_score.toUpperCase()}</span>
            </div>
            ${report.warnings.length > 0 ? `
                <div style="margin-top: 10px; color: var(--warning); font-size: 0.8rem;">
                    <strong>Warnings:</strong>
                    <ul style="margin-left: 15px;">
                        ${report.warnings.map(w => `<li>${w}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
    }
    
    resetBtn.classList.remove('hidden');
}

// Event Listeners
startBtn.addEventListener('click', startScan);
resetBtn.addEventListener('click', () => {
    window.location.reload();
});

// Start camera on load
initCamera();
