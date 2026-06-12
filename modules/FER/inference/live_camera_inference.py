#!/usr/bin/env python3
"""
Face Emotion Recognition — Live Camera (VSCode / PC)
=====================================================
Requirements:
    pip install torch torchvision opencv-python Pillow

Usage:
    python live_camera_inference.py
    python live_camera_inference.py --model_dir ./emotion_model --camera 0

Keys while running:
    q   quit
    s   save screenshot
    p   pause / resume

Yoga-pose integration tip
─────────────────────────
Call get_yoga_poses(emotion) to map the detected emotion to
yoga recommendations in your downstream module.
"""
import cv2, torch, json, argparse, time
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

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


# ── Emotion → yoga-pose suggestions (wire to your module) ─────
YOGA_MAP = {
    "Fear"     : ["Child's Pose (Balasana)", "Mountain Pose (Tadasana)", "Tree Pose (Vrksasana)"],
    "Happiness": ["Sun Salutation (Surya Namaskar)", "Warrior I (Virabhadrasana I)", "Camel Pose"],
    "Sadness"  : ["Heart-Opening (Ustrasana)", "Warrior II", "Bridge Pose (Setu Bandhasana)"],
    "Anger"    : ["Child's Pose (Balasana)", "Forward Fold (Uttanasana)", "Corpse Pose (Savasana)"],
    "Neutral"  : ["Mountain Pose", "Seated Meditation", "Equal-Standing Pose"],
}

def get_yoga_poses(emotion: str) -> list[str]:
    return YOGA_MAP.get(emotion, ["Mountain Pose (Tadasana)"])


# ── Colours (BGR) per emotion ─────────────────────────────────
COLORS = {
    "Fear"     : (0, 0, 220),
    "Happiness": (0, 200, 0),
    "Sadness"  : (220, 100, 0),
    "Anger"    : (0, 0, 180),
    "Neutral"  : (160, 160, 160),
}


class EmotionPredictor:
    def __init__(self, model_dir: str, device: str | None = None):
        self.dir    = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Device: {self.device}")

        with open(self.dir / "model_config.json") as f:
            cfg = json.load(f)

        self.idx2em  = {int(k): v for k, v in cfg["idx_to_emotion"].items()}
        self.n_cls   = cfg["num_classes"]
        self.imsz    = cfg["image_size"]

        self.model = EmotionDenseNet(self.n_cls).to(self.device)
        ckpt       = torch.load(self.dir / "best_model.pth", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tfm = transforms.Compose([
            transforms.Resize((self.imsz, self.imsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

        self.face_det = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # warm-up
        with torch.no_grad():
            self.model(torch.zeros(1,3,self.imsz,self.imsz,device=self.device))
        print("[INFO] Model ready.")

    @torch.no_grad()
    def predict(self, face_bgr: np.ndarray):
        rgb    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t      = self.tfm(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        probs  = torch.softmax(self.model(t), 1)[0].cpu().numpy()
        idx    = int(probs.argmax())
        return self.idx2em[idx], float(probs[idx]), probs

    def overlay(self, frame, x, y, w, h, em, conf, probs):
        col  = COLORS.get(em, (200,200,200))
        cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)

        label = f"{em}: {conf*100:.1f}%"
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.68,2)
        cv2.rectangle(frame,(x,y-th-12),(x+tw+6,y),col,-1)
        cv2.putText(frame,label,(x+3,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.68,(255,255,255),2)

        # mini bar chart top-left
        for i,(k,name) in enumerate(self.idx2em.items()):
            by = 30 + i*30
            bw = int(probs[k]*160)
            cv2.rectangle(frame,(8,by),(8+bw,by+20),COLORS.get(name,(180,180,180)),-1)
            cv2.putText(frame,f"{name[:3]} {probs[k]*100:.0f}%",(175,by+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)

        # yoga suggestion (first 2)
        poses = get_yoga_poses(em)
        cv2.putText(frame,f"Yoga: {poses[0]}",(x,y+h+18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,220,220),1)
        return frame


def main(model_dir: str, camera_id: int = 0):
    pred = EmotionPredictor(model_dir)
    cap  = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    fps, t0, fc, paused = 0, time.time(), 0, False
    print("Keys: q=quit  s=screenshot  p=pause")

    while True:
        ok, frame = cap.read()
        if not ok: break
        if paused:
            cv2.putText(frame,"PAUSED",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
            cv2.imshow("Emotion Recognition",frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord("p"): paused = False
            continue

        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = pred.face_det.detectMultiScale(gray,1.1,5,minSize=(60,60))
        for (x,y,w,h) in faces:
            roi = frame[y:y+h,x:x+w]
            if roi.size == 0: continue
            em, conf, probs = pred.predict(roi)
            pred.overlay(frame,x,y,w,h,em,conf,probs)

        fc += 1
        if (dt := time.time()-t0) >= 1.0:
            fps, fc, t0 = round(fc/dt), 0, time.time()

        cv2.putText(frame,f"FPS:{fps}",(frame.shape[1]-100,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.68,(0,255,255),2)
        cv2.imshow("Emotion Recognition",frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        elif k == ord("s"):
            n = f"shot_{int(time.time())}.jpg"; cv2.imwrite(n,frame); print(f"Saved {n}")
        elif k == ord("p"): paused = True

    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./emotion_model")
    ap.add_argument("--camera",    type=int, default=0)
    a  = ap.parse_args()
    main(a.model_dir, a.camera)
