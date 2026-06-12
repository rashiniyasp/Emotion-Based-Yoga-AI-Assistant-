# MatNODE Yoga-82 to Yoga-16

This project fine-tunes the supplied `MatNODE_Yoga82.pth` checkpoint on the
local Yoga-16 image dataset. It uses the MediaPipe **Tasks API**
(`PoseLandmarker`) and never uses `mp.solutions`.

## Completed Results

- Yoga-16 source data: 896 train, 128 validation, 256 test images
- MediaPipe detections: 895 train, 128 validation, 256 test
- Selected checkpoint: epoch 22 by lowest validation loss
- Selected validation accuracy: 92.97%
- Highest observed validation accuracy: 93.75%
- Final untouched Yoga-16 test accuracy: **94.53%**
- Final test macro-F1: **0.9444**
- Yoga-16 parameters: 66,992
- Deployment complexity: 6.053 MMAC / 12.106 MFLOP per sample
- Cached-feature CPU inference: mean 0.579 ms per sample
- ONNX size: 368,504 bytes; ONNX Runtime verification passed

The model-only timing excludes MediaPipe landmark extraction. Full camera/video
latency is displayed by `inference.py`.

## Architecture And Features

The checkpoint-compatible model remains:

- 16 Y-axis rotations from -180 through 180 degrees
- 212 features per view
- 99 landmark coordinates (33 x 3)
- 8 normalized 3D joint angles
- 105 bone-vector values (35 x 3)
- 48-dimensional encoder
- Neural ODE with 64 hidden units and Softplus
- Two 4-head transformer encoder layers
- Mean pooling and classification head

PyTorch training/inference uses adaptive `dopri5`, matching the supplied code.
ONNX uses eight fixed RK4 steps because adaptive `torchdiffeq` execution is not
portable to ONNX/Raspberry Pi. The exported logits match the PyTorch RK4 wrapper
within `7.15e-7` maximum absolute error.

## Label Mapping

| Index | Pose |
|---:|---|
| 0 | chair_pose |
| 1 | dolphin_plank_pose |
| 2 | downward-facing_dog_pose |
| 3 | fish_pose |
| 4 | goddess_pose |
| 5 | locust_pose |
| 6 | lord_of_the_dance_pose |
| 7 | low_lunge_pose |
| 8 | seated_forward_bend_pose |
| 9 | side_plank_pose |
| 10 | staff_pose |
| 11 | tree_pose |
| 12 | warrior_1_pose |
| 13 | warrior_2_pose |
| 14 | warrior_3_pose |
| 15 | wide-angle_seated_forward_bend_pose |

The machine-readable mapping is in `labels_yoga16.json` and is also embedded in
the Yoga-16 checkpoint.

## Files

- `model.py`: original model, transfer checkpoint loader, ONNX RK4 wrapper
- `pose_features.py`: MediaPipe Tasks extraction and 212-feature generation
- `prepare_dataset.py`: landmark caching and failed-detection manifest
- `yoga_data.py`: Yoga-16 cache and legacy Yoga-82 NPY loaders
- `train_yoga16.py`: validation-selected transfer learning
- `evaluate.py`: metrics, predictions, confusion matrix, timing, FLOPs/MACs
- `inference.py`: live camera or video-file recognition
- `export_onnx.py`: ONNX export and ONNX Runtime verification
- `PROJECT_REPORT.json`: run summary and artifact index

## Reproduce

PowerShell commands from this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_dataset.py
.\.venv\Scripts\python.exe train_yoga16.py --epochs 50 --patience 10
.\.venv\Scripts\python.exe evaluate.py --checkpoint artifacts\MatNODE_Yoga16.pth --dataset-type yoga16 --output-dir artifacts\yoga16_test
.\.venv\Scripts\python.exe export_onnx.py
```

The official Pose Landmarker Lite task file is already at
`models/pose_landmarker_lite.task`.

## Camera And Video

Camera 0:

```powershell
.\.venv\Scripts\python.exe inference.py --source 0
```

Video file, with annotated output:

```powershell
.\.venv\Scripts\python.exe inference.py --source input.mp4 --output-video artifacts\annotated.mp4
```

Raspberry Pi / ONNX Runtime:

```powershell
.\.venv\Scripts\python.exe inference.py --source 0 --onnx-model artifacts\MatNODE_Yoga16.onnx
```

Press `q` to close the display.

## Yoga-82 Final Test

The supplied workspace contains the Yoga-82 checkpoint but no Yoga-82 test
folder, so real Yoga-82 accuracy/F1 cannot be calculated without inventing
data. The checkpoint itself was profiled in
`artifacts/yoga82_checkpoint_profile.json`.

When the original skeleton test folder is available, run:

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --checkpoint MatNODE_Yoga82.pth `
  --dataset-type legacy-npy `
  --dataset-root D:\path\to\Yoga_82_Balanced_2026 `
  --split test `
  --output-dir artifacts\yoga82_test
```

Expected layout:

```text
Yoga_82_Balanced_2026/
  test/
    pose_class_name/
      sample.npy
```

For the plain Yoga-82 checkpoint, class indices are inferred from alphabetically
sorted test-folder names, matching the original dataset code. The generated
`results.json` will contain that complete index-to-pose mapping.

## Saved Artifacts

- `artifacts/MatNODE_Yoga16.pth`: best validation-selected model
- `artifacts/MatNODE_Yoga16.history.json`: all 32 training epochs
- `artifacts/MatNODE_Yoga16.onnx`: Raspberry Pi deployment model
- `artifacts/MatNODE_Yoga16.onnx.json`: ONNX verification
- `artifacts/yoga16_test/results.json`: final metrics and per-class report
- `artifacts/yoga16_test/predictions.csv`: every test prediction and latency
- `artifacts/yoga16_test/confusion_matrix.csv`: 16 x 16 confusion matrix
- `artifacts/yoga82_checkpoint_profile.json`: Yoga-82 size/timing/complexity

One training image had no detectable pose:
`wide-angle_seated_forward_bend_pose/15.jpg`. It is recorded in the training
manifest and was excluded; no validation or test image was excluded.
