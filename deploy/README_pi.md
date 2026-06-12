# Raspberry Pi Deployment Guide

## Prerequisites

- Raspberry Pi 4 (2GB+ RAM) or Raspberry Pi 5
- Raspberry Pi OS (64-bit Bookworm recommended)
- USB webcam or Pi Camera Module v2/v3
- HDMI-connected monitor
- Python 3.11+

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/emotion-aware-yoga-system.git ~/yoga-system
cd ~/yoga-system

# Run the installer
chmod +x deploy/install_pi.sh
./deploy/install_pi.sh

# Start the application
source venv/bin/activate
python deploy/pi_app.py
```

## Model Files

The following model files must be present (not included in git — too large):

| File | Size | Source |
|------|------|--------|
| `models/fer/best_model.pth` | ~91 MB | Training output |
| `models/fer/emotion_model.onnx` | ~30 MB | ONNX export |
| `models/matnode/MatNODE_Yoga16.pth` | ~283 KB | Training output |
| `models/matnode/MatNODE_Yoga16.onnx` | ~369 KB | ONNX export |
| `models/acorn/jcat_best.pth` | ~427 KB | Training output |
| `models/acorn/train_data.npz` | ~83 KB | Training output |
| `models/mediapipe/pose_landmarker_lite.task` | ~5.8 MB | [Download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task) |

Copy these from your training machine:
```bash
scp -r models/ pi@raspberrypi:~/yoga-system/models/
```

## Controls

| Key | Action |
|-----|--------|
| `Space` | Capture face (M1) / Confirm (M1→M2) |
| `R` | Reset current module |
| `Esc` | Go back one step |
| `Q` | Quit |

## Auto-Start on Boot

```bash
sudo systemctl enable yoga-assistant
sudo systemctl start yoga-assistant

# Check status
sudo systemctl status yoga-assistant

# View logs
journalctl -u yoga-assistant -f
```

## Performance Notes

- **FER (ONNX)**: ~200ms per face crop on Pi 4
- **MAtNODE (ONNX)**: ~50ms per frame on Pi 4
- **ACORN (50 steps)**: ~500ms per correction on Pi 4
- **MediaPipe Lite**: ~30ms per frame on Pi 4
- **Total pipeline**: ~15-20 FPS excluding ACORN (ACORN runs in background thread)

## Troubleshooting

### Camera not detected
```bash
# Check USB camera
ls /dev/video*

# Test with OpenCV
python3 -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened()); c.release()"

# For Pi Camera Module, ensure it's enabled:
sudo raspi-config  # → Interface Options → Camera → Enable
```

### Display issues
```bash
# Ensure DISPLAY is set
export DISPLAY=:0

# If running via SSH, use X forwarding:
ssh -X pi@raspberrypi
```

### Memory issues
```bash
# Increase GPU memory split (for camera)
sudo raspi-config  # → Performance Options → GPU Memory → 128

# Monitor memory usage
htop
```
