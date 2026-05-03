# Active Vision
This project aims to develop an RL agent to determine the best viewing angle to reduce ambiguity for vision based tasks.

## Sample use for shapenet_gym
```
env = shapenet_gym.env.ShapeNetViewEnv(
    dataset_root="data\ShapeNetCore",
    max_steps=20,
    image_size=(224, 224),
    theta_step_deg=15.0,
    phi_step_deg=15.0,
    offscreen=False
)
```

## Setup

### 1. Install dependencies

```bash
pip install stable-baselines3 ultralytics gymnasium trimesh pyrender \
            opencv-python torch torchvision \
            "PyOpenGL>=3.1.7" PyOpenGL-accelerate \
            nltk
python -c "import nltk; nltk.download('wordnet')"   # for ShapeNet/ImageNet correctness mapping
# optional, for live training dashboards
pip install wandb tensorboard
```

### 2. Download the dataset

ShapeNetCore is gated on HuggingFace. Accept the dataset terms at
https://huggingface.co/datasets/ShapeNet/ShapeNetCore while logged in, then:

```bash
huggingface-cli login   # paste your HF token
mkdir -p data/ShapeNetCore

# Two categories (~9 GB) — enough to start training
huggingface-cli download ShapeNet/ShapeNetCore \
    02691156.zip 02958343.zip \
    --repo-type dataset --local-dir data/ShapeNetCore

# OR the full dataset (~24 GB)
huggingface-cli download ShapeNet/ShapeNetCore \
    --repo-type dataset --local-dir data/ShapeNetCore
```

### 3. Extract

```bash
python extract_shapenet.py --root data/ShapeNetCore
python diagnose_dataset.py --root data/ShapeNetCore   # sanity-check layout
```

## Training

The script `train_ppo.py` runs PPO with a YOLOv8-cls entropy-reduction reward.
By default it trains on airplanes (`02691156`) and cars (`02958343`); pass
`--categories <synset_id ...>` to change.

### macOS (Apple Silicon, MPS GPU)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_ppo.py --max_steps 50 --total_timesteps 100000
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` lets unsupported ops fall back to CPU silently
instead of crashing. Native CGL is used for offscreen rendering — no extra GL
setup needed.

### Linux (headless servers, CPU or CUDA)

```bash
# CPU-only headless box
PYOPENGL_PLATFORM=osmesa python train_ppo.py --max_steps 50 --total_timesteps 100000

# NVIDIA GPU box
PYOPENGL_PLATFORM=egl python train_ppo.py --max_steps 50 --total_timesteps 100000
```

You'll need `libOSMesa` (`apt install libosmesa6-dev`) or a working EGL/CUDA
stack. PyTorch automatically picks `cuda` when available.

### Windows

```powershell
python train_ppo.py --max_steps 50 --total_timesteps 100000
```

Pyrender uses the native WGL context for offscreen rendering. PyTorch will use
CUDA automatically if installed.

### Common flags

```
--max_steps              Episode length (default: 50)
--total_timesteps        Total training steps (default: 100000)
--categories             Synset IDs to train on (default: 02691156 02958343).
                         Omit to use every synset folder under data/ShapeNetCore.
--upper_hemisphere_only  Restrict the camera to theta >= 0 (skip looking up
                         from below the object). Match this between train + eval.
--split                  Object-instance split: train | val | test | all.
                         Default: train. Splits are deterministic md5 buckets
                         on (synset_id, model_id) — 70/15/15.
--use_wandb              Log to Weights & Biases (requires `pip install wandb tensorboard`)
--save_dir               Where to save the trained model (default: outputs/ppo)
```

Trained models land in `outputs/ppo/ppo_maxsteps{N}.zip`.

## Evaluation

```bash
python evaluate.py --model_path outputs/ppo/ppo_maxsteps50.zip \
                   --upper_hemisphere_only      # match training
                   # --split test               # default; use val during dev
```

Writes `outputs/eval/<run>.json` and a markdown summary with top-1 accuracy,
AUCC, and per-category breakdown. Correctness uses the WordNet hypernym
mapping in `shapenet_gym/labels.py` (`SYNSET_TO_TARGET`).

## Project layout

```
shapenet_gym/        Gymnasium environment, renderer, dataset loader, rewards
train_ppo.py         PPO training script
extract_shapenet.py  Unzip downloaded ShapeNetCore archives
diagnose_dataset.py  Verify dataset layout
yolov8n-cls.pt       Pretrained YOLOv8 nano classifier (reward signal)
main.ipynb           Proof-of-concept: viewpoint affects classifier confidence
```
