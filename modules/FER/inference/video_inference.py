#!/usr/bin/env python3
"""
Face Emotion Recognition — Video File Inference
================================================
Usage:
    python video_inference.py --video input.mp4 --model_dir ./emotion_model
    python video_inference.py --video input.mp4 --save output_annotated.mp4
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


COLORS = {
    "Fear":(0,0,220),"Happiness":(0,200,0),
    "Sadness":(220,100,0),"Anger":(0,0,180),"Neutral":(160,160,160),
}

class VideoEmotionInference:
    def __init__(self, model_dir, device=None):
        self.dir    = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(self.dir/"model_config.json") as f: cfg = json.load(f)
        self.idx2em = {int(k):v for k,v in cfg["idx_to_emotion"].items()}
        self.imsz   = cfg["image_size"]
        m = EmotionDenseNet(cfg["num_classes"]).to(self.device)
        ck = torch.load(self.dir/"best_model.pth", map_location=self.device)
        m.load_state_dict(ck["model_state_dict"]); m.eval(); self.model = m
        self.tfm = transforms.Compose([
            transforms.Resize((self.imsz,self.imsz)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.det = cv2.CascadeClassifier(
            cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

    @torch.no_grad()
    def predict(self, face_bgr):
        rgb  = cv2.cvtColor(face_bgr,cv2.COLOR_BGR2RGB)
        t    = self.tfm(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.model(t),1)[0].cpu().numpy()
        idx  = int(prob.argmax())
        return self.idx2em[idx], float(prob[idx]), prob

    def annotate(self, frame, x, y, w, h, em, conf):
        col = COLORS.get(em,(200,200,200))
        cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
        lbl = f"{em}: {conf*100:.1f}%"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
        cv2.rectangle(frame,(x,y-th-10),(x+tw+4,y),col,-1)
        cv2.putText(frame,lbl,(x+2,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)

    def process(self, video_path, save_path=None, show=True):
        cap = cv2.VideoCapture(video_path)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if save_path:
            writer = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

        frame_idx, t0 = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1
            gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = self.det.detectMultiScale(gray,1.1,5,minSize=(60,60))
            for (fx,fy,fw,fh) in faces:
                roi = frame[fy:fy+fh,fx:fx+fw]
                if roi.size==0: continue
                em,conf,_ = self.predict(roi)
                self.annotate(frame,fx,fy,fw,fh,em,conf)
            if writer: writer.write(frame)
            if show:
                cv2.imshow("Emotion — Video",frame)
                if cv2.waitKey(1)&0xFF==ord("q"): break
            if frame_idx % 50 == 0:
                pct = frame_idx/total*100 if total>0 else 0
                print(f"  {frame_idx}/{total} ({pct:.1f}%)  {time.time()-t0:.1f}s elapsed")

        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅  Done  {frame_idx} frames  ({time.time()-t0:.1f}s)")
        if save_path: print(f"   Saved → {save_path}")


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",     required=True)
    ap.add_argument("--model_dir", default="./emotion_model")
    ap.add_argument("--save",      default=None, help="output video path")
    ap.add_argument("--no_show",   action="store_true")
    a  = ap.parse_args()
    inf = VideoEmotionInference(a.model_dir)
    inf.process(a.video, a.save, show=not a.no_show)
