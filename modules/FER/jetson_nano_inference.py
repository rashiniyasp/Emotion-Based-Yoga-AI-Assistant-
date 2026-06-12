#!/usr/bin/env python3
"""
Face Emotion Recognition — Jetson Nano Inference
=================================================
Two modes:
  --mode pytorch   Uses best_model.pth  (recommended for JetPack ≥ 4.6)
  --mode onnx      Uses emotion_model.onnx via ONNX Runtime

Camera options:
  --camera usb       USB/V4L2 camera  (default, /dev/video0)
  --camera csi       CSI IMX219 camera via GStreamer pipeline

Setup on Jetson Nano
────────────────────
  # PyTorch (from NVIDIA wheel, JetPack 4.6):
  pip3 install --upgrade pip
  pip3 install Cython numpy
  pip3 install https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch/torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl
  pip3 install torchvision==0.11.0

  # OpenCV is usually pre-installed; if not:
  sudo apt-get install python3-opencv

  # For ONNX mode:
  pip3 install onnxruntime  (CPU) — or build onnxruntime-gpu from source

  # Copy these files to Jetson:
  rsync -av emotion_model/ jetson@<IP>:~/emotion_model/
  scp jetson_nano_inference.py jetson@<IP>:~/

Usage:
  python3 jetson_nano_inference.py --mode pytorch --camera usb
  python3 jetson_nano_inference.py --mode onnx    --camera csi
"""
import cv2, json, time, argparse
import numpy as np
from pathlib import Path

# ── TensorRT-style latency measurement ───────────────────────
class Timer:
    def __init__(self): self.t = time.perf_counter()
    def elapsed_ms(self): return (time.perf_counter()-self.t)*1000


# ── GStreamer pipeline for CSI (IMX219) camera ──────────────
def gst_pipeline(sensor_id=0, w=640, h=480, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},"
        f"format=NV12,framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw,width={w},height={h},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink"
    )


# ── Shared pre-processing (numpy, no torchvision) ──────────
MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
STD  = np.array([0.229,0.224,0.225],dtype=np.float32)

def preprocess(face_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    img  = cv2.resize(face_bgr,(size,size))
    img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    img  = (img - MEAN) / STD
    return img.transpose(2,0,1)[None].astype(np.float32)     # NCHW


COLORS = {
    "Fear":(0,0,220),"Happiness":(0,200,0),
    "Sadness":(220,100,0),"Anger":(0,0,180),"Neutral":(160,160,160),
}

YOGA_MAP = {
    "Fear"     : ["Child Pose","Mountain Pose"],
    "Happiness": ["Sun Salutation","Warrior I"],
    "Sadness"  : ["Heart-Opening","Bridge Pose"],
    "Anger"    : ["Child Pose","Forward Fold"],
    "Neutral"  : ["Mountain Pose","Seated Meditation"],
}


# ═══════════════════════════════════════════════════════════
# PyTorch backend
# ═══════════════════════════════════════════════════════════

import torch, torch.nn as nn
from torchvision import models

class EmotionDenseNet(nn.Module):
    """DenseNet-121 — must match training architecture exactly."""
    def __init__(self, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        bb               = models.densenet121(weights=None)
        in_feat          = bb.classifier.in_features
        bb.classifier    = nn.Sequential(
            nn.BatchNorm1d(in_feat), nn.Dropout(dropout),
            nn.Linear(in_feat, 512), nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),     nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes),
        )
        self.net = bb
    def forward(self, x): return self.net(x)


def load_pytorch(model_dir, device):
    import torch
    cfg = json.load(open(Path(model_dir)/"model_config.json"))
    m   = EmotionDenseNet(cfg["num_classes"]).to(device)
    ck  = torch.load(Path(model_dir)/"best_model.pth", map_location=device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    idx2em = {int(k):v for k,v in cfg["idx_to_emotion"].items()}
    imsz   = cfg["image_size"]

    def predict(face_bgr):
        import torch
        x     = torch.from_numpy(preprocess(face_bgr,imsz)).to(device)
        with torch.no_grad():
            prob = torch.softmax(m(x),1)[0].cpu().numpy()
        idx  = int(prob.argmax())
        return idx2em[idx], float(prob[idx])

    return predict, imsz


# ═══════════════════════════════════════════════════════════
# ONNX Runtime backend
# ═══════════════════════════════════════════════════════════
def load_onnx(model_dir):
    import onnxruntime as ort
    cfg  = json.load(open(Path(model_dir)/"model_config.json"))
    sess = ort.InferenceSession(
        str(Path(model_dir)/"emotion_model.onnx"),
        providers=["CUDAExecutionProvider","CPUExecutionProvider"],
    )
    idx2em = {int(k):v for k,v in cfg["idx_to_emotion"].items()}
    imsz   = cfg["image_size"]
    in_nm  = sess.get_inputs()[0].name

    def predict(face_bgr):
        x    = preprocess(face_bgr, imsz)
        out  = sess.run(None,{in_nm:x})[0][0]
        prob = np.exp(out)/np.exp(out).sum()
        idx  = int(prob.argmax())
        return idx2em[idx], float(prob[idx])

    return predict, imsz


# ═══════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════
def run(args):
    device = "cuda" if (args.mode=="pytorch" and
                        __import__("torch").cuda.is_available()) else "cpu"

    if args.mode == "pytorch":
        predict, imsz = load_pytorch(args.model_dir, device)
        print(f"[INFO] PyTorch backend  device={device}")
    else:
        predict, imsz = load_onnx(args.model_dir)
        print("[INFO] ONNX Runtime backend")

    if args.camera == "csi":
        src = gst_pipeline(flip=args.flip)
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        print("[INFO] CSI camera via GStreamer")
    else:
        cap = cv2.VideoCapture(args.device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        print(f"[INFO] USB/V4L2 camera  id={args.device_id}")

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    det    = cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    fps, t0, fc = 0, time.time(), 0

    print("Keys: q=quit  s=screenshot")
    while True:
        ok, frame = cap.read()
        if not ok: break

        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray,1.1,5,minSize=(60,60))

        for (x,y,w,h) in faces:
            roi = frame[y:y+h,x:x+w]
            if roi.size==0: continue
            tmr = Timer()
            em, conf = predict(roi)
            lat_ms   = tmr.elapsed_ms()

            col = COLORS.get(em,(200,200,200))
            cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
            lbl = f"{em}: {conf*100:.1f}%  ({lat_ms:.0f}ms)"
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(frame,(x,y-th-10),(x+tw+4,y),col,-1)
            cv2.putText(frame,lbl,(x+2,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            # yoga hint
            poses = YOGA_MAP.get(em,[])
            if poses:
                cv2.putText(frame,f"Yoga: {poses[0]}",(x,y+h+18),
                            cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,220,220),1)

        fc += 1
        if (dt:=time.time()-t0)>=1.0:
            fps, fc, t0 = round(fc/dt), 0, time.time()
        cv2.putText(frame,f"FPS:{fps}",(frame.shape[1]-100,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)

        cv2.imshow("Emotion — Jetson Nano", frame)
        k = cv2.waitKey(1)&0xFF
        if k==ord("q"): break
        elif k==ord("s"):
            n=f"shot_{int(time.time())}.jpg"; cv2.imwrite(n,frame); print(f"Saved {n}")

    cap.release(); cv2.destroyAllWindows()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./emotion_model")
    ap.add_argument("--mode",      choices=["pytorch","onnx"], default="pytorch")
    ap.add_argument("--camera",    choices=["usb","csi"], default="usb")
    ap.add_argument("--device_id", type=int, default=0, help="V4L2 device id for USB cam")
    ap.add_argument("--flip",      type=int, default=0, help="GStreamer flip-method (CSI cam)")
    a  = ap.parse_args()
    run(a)
