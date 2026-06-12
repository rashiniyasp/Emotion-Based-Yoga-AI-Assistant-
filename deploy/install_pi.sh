#!/bin/bash
# install_pi.sh — Automated installation for Raspberry Pi
# Run: chmod +x install_pi.sh && ./install_pi.sh

set -e

echo "============================================"
echo "  Yoga Assistant — Raspberry Pi Installer"
echo "============================================"
echo ""

INSTALL_DIR="${HOME}/yoga-system"

# 1. System dependencies
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-dev python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libatlas-base-dev libjasper-dev libhdf5-dev

# 2. Clone or update repo
echo "[2/6] Setting up project directory..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  Directory exists. Updating..."
    cd "$INSTALL_DIR"
    git pull 2>/dev/null || echo "  Not a git repo — skipping pull."
else
    echo "  Creating fresh install at $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    # If running from the repo, copy files
    if [ -f "$(dirname $0)/../app.py" ]; then
        cp -r "$(dirname $0)/../"* "$INSTALL_DIR/"
    else
        echo "  Please clone the repo into $INSTALL_DIR manually."
    fi
fi
cd "$INSTALL_DIR"

# 3. Virtual environment
echo "[3/6] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
echo "[4/6] Installing Python packages..."
pip install --upgrade pip wheel setuptools
pip install -r deploy/pi_requirements.txt

# Install PyTorch CPU for ARM
echo "  Installing PyTorch (CPU-only for ARM)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    echo "  PyTorch ARM wheel not found. Install manually from piwheels."

# 5. Download model files (if not present)
echo "[5/6] Checking model files..."
declare -A MODEL_FILES=(
    ["models/mediapipe/pose_landmarker_lite.task"]="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

for filepath in "${!MODEL_FILES[@]}"; do
    if [ ! -f "$filepath" ]; then
        echo "  Downloading $filepath..."
        mkdir -p "$(dirname $filepath)"
        wget -q -O "$filepath" "${MODEL_FILES[$filepath]}" 2>/dev/null || \
            echo "  Could not download $filepath. Please download manually."
    else
        echo "  ✓ $filepath exists"
    fi
done

# Check for other model files
for f in models/fer/best_model.pth models/matnode/MatNODE_Yoga16.pth models/acorn/jcat_best.pth; do
    if [ -f "$f" ]; then
        echo "  ✓ $f exists"
    else
        echo "  ✗ MISSING: $f — please copy from your training machine"
    fi
done

# 6. Configure systemd service
echo "[6/6] Setting up auto-start service..."
sudo tee /etc/systemd/system/yoga-assistant.service > /dev/null << EOF
[Unit]
Description=Yoga Assistant System
After=graphical.target

[Service]
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python deploy/pi_app.py
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=XAUTHORITY=$HOME/.Xauthority

[Install]
WantedBy=graphical.target
EOF

sudo systemctl daemon-reload
echo "  Service created. To enable auto-start on boot:"
echo "    sudo systemctl enable yoga-assistant"
echo "  To start manually:"
echo "    sudo systemctl start yoga-assistant"

echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "  To run: cd $INSTALL_DIR && source venv/bin/activate && python deploy/pi_app.py"
echo "  Or:     sudo systemctl start yoga-assistant"
echo ""

# Test camera access
echo "Testing camera access..."
python3 -c "import cv2; c=cv2.VideoCapture(0); print('Camera:', 'OK' if c.isOpened() else 'FAIL'); c.release()"
