# Face Emotion Recognition — DenseNet-121

Trained on RAF-DB (5 emotions): **Fear · Happiness · Sadness · Anger · Neutral**

## Model details
| Item           | Value                       |
|----------------|-----------------------------|
| Backbone       | DenseNet-121 (ImageNet pre-trained) |
| Loss           | CrossEntropy + label smoothing, class-balanced weights |
| Optimiser      | AdamW                       |
| Scheduler      | Cosine Annealing LR         |
| Early stopping | patience = 12              |
| Test accuracy  | see classification_report.txt |
| Input size     | 224 × 224 RGB               |
| ONNX opset     | 11                          |

## Files
```
emotion_model/
├── best_model.pth              ← PyTorch weights (best val acc)
├── final_model.pth             ← PyTorch weights (last epoch)
├── emotion_model.onnx          ← ONNX (Jetson / TensorRT)
├── model_config.json           ← class maps, normalisation
├── label_encoder.json          ← index ↔ emotion string
├── class_distribution.png
├── training_curves.png
├── confusion_matrix.png
├── classification_report.txt
├── live_camera_inference.py    ← VSCode live camera
├── video_inference.py          ← VSCode video file
└── jetson_nano_inference.py    ← Jetson Nano (USB / CSI cam)
```

## Quick start — VSCode live camera
```bash
pip install -r requirements.txt
python live_camera_inference.py --model_dir ./emotion_model --camera 0
```

## Quick start — video file
```bash
python video_inference.py --video input.mp4 --model_dir ./emotion_model --save output.mp4
```

## Quick start — Jetson Nano (USB camera)
```bash
# install PyTorch from NVIDIA wheel first (see requirements.txt)
python3 jetson_nano_inference.py --mode pytorch --camera usb
# for ONNX mode:
python3 jetson_nano_inference.py --mode onnx    --camera usb
```

## TensorRT (optional, max speed on Jetson)
```bash
# Convert ONNX → TRT engine (run ON the Jetson)
python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser  = trt.OnnxParser(network, logger)
with open('emotion_model/emotion_model.onnx','rb') as f:
    parser.parse(f.read())
config = builder.create_builder_config()
config.max_workspace_size = 1 << 28  # 256 MB
engine_bytes = builder.build_serialized_network(network, config)
with open('emotion_model/emotion_trt.engine','wb') as f:
    f.write(engine_bytes)
"
```

## Yoga integration
Each inference script exports `get_yoga_poses(emotion) -> list[str]`.
Wire this to your yoga-recommendation module:
```python
from live_camera_inference import get_yoga_poses
poses = get_yoga_poses("Sadness")
# → ["Heart-Opening (Ustrasana)", "Warrior II", "Bridge Pose"]
```
