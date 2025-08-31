# Image Captioning — BLIP (Fine‑Tuning & Evaluation)

> Lightweight, pragmatic fine‑tuning and evaluation pipeline around **Salesforce BLIP** for image captioning. Uses Hugging Face **Transformers** and **Datasets**, with a clean modular layout and a single‑file inference helper.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/transformers-hf-green" alt="Transformers">
  <img src="https://img.shields.io/badge/torch-gpu%20recommended-lightgrey" alt="torch">
  <img src="https://img.shields.io/badge/status-experimental-orange" alt="status">
</p>

---

## Why this repo

- **End‑to‑end**: Train ➜ evaluate ➜ export ➜ run inference with a couple of functions.
- **Sane defaults**: BLIP base model, Flickr8k by default; optional COCO‑Karpathy split.
- **Minimal surface area**: A small, readable codebase that’s straightforward to extend or productionize.
- **Metrics you actually care about**: BLEU during training, COCO‑style metrics (CIDEr/METEOR/SPICE) post‑training.

If you’re looking for a “batteries included” library, see LAVIS. If you want a focused, no‑frills training script that you can own and reason about, you’re in the right place.

---

## TL;DR – quick start

```bash
# 1) Create and activate an environment
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA/OS
pip install transformers datasets evaluate tensorboard sacrebleu pillow numpy
# For COCO metrics (optional; needed for CIDEr, METEOR, SPICE):
pip install pycocotools pycocoevalcap

# 3) Train (defaults: Flickr8k, BLIP base)
python -m blip_caption.main_train
```

> **Note**: COCO metrics run after training on the validation split and are written to `blip-open-out/coco_metrics.json`. If SPICE throws a Java error, see **Troubleshooting** below.

---

## Project structure

```
blip_caption/
├─ __init__.py
├─ config.py           # All defaults live here (model paths, data, training, generation, metrics)
├─ data.py             # Loads Flickr8k or COCO‑Karpathy via 🤗 Datasets, builds train/val/test
├─ collator.py         # Batch collation using BlipProcessor; builds labels with pad masked to -100
├─ modeling.py         # Model + processor builders (BLIP base by default); optional vision freeze
├─ training.py         # HF TrainingArguments + Trainer wiring
├─ metrics.py          # BLEU (training), COCO metrics (post‑train)
├─ inference.py        # Single‑image caption helper
└─ main_train.py       # End‑to‑end train→evaluate→export entrypoint
```

### Defaults worth knowing

`blip_caption/config.py` is a set of `@dataclass` blocks you can treat like a single config object:

- **Model & outputs**
  - `paths.model_id`: `"Salesforce/blip-image-captioning-base"`
  - `paths.output_dir`: `"blip-open-out"`

- **Data**
  - `data.use_dataset`: `"flickr8k"` or `"coco_karpathy"`
  - Flickr8k HF id: `ariG23498/flickr8k`
  - COCO‑Karpathy HF id: `yerevann/coco-karpathy`
  - Repro seed: `42`
  - Flickr8k split recipe: 90% train / 5% val / 5% test (random, deterministic by seed)

- **Training**
  - Epochs: `4`, LR: `5e-5`, per‑device batch size: `8` (train & eval), gradient accumulation: `2`
  - Gradient checkpointing: `True`
  - Freeze vision encoder: `False` (set `True` for lighter fine‑tuning)
  - Logging every `50` steps, keep `2` checkpoints
  - Early selection by `"sacrebleu"` best metric

- **Generation (inference/eval)**
  - `max_txt_len`: `40`, `gen_max_new_tokens`: `30`, `num_beams`: `5`, `length_penalty`: `1.0`, `early_stopping`: `True`

- **COCO metrics**
  - `do_spice`: `True` (requires Java), `spice_subsample`: `1500` (speeds things up), JSON output enabled

> **Pro tip**: No CLI yet—tweak anything in `config.py` (or copy the dataclasses into your own script and pass a custom config into the builders).

---

## Training & evaluation

### 1) Train
`main_train.py` does the heavy lifting:
- Sets seeds for reproducibility.
- Loads the dataset according to `data.use_dataset`.
- Builds processor + model (with optional vision freezing).
- Wires an HF `Trainer` with a custom collator and a BLEU compute function.
- Trains and prints the **best BLEU** seen during training.

Run:
```bash
python -m blip_caption.main_train
```

### 2) COCO‑style metrics (post‑train)
After training, `main_train.py` decodes validation captions and computes:
- **CIDEr**, **METEOR**, **SPICE** (optional), and **BLEU‑4** summary
- Writes a compact `blip-open-out/coco_metrics.json`

> SPICE can be slow and needs Java; set `coco.do_spice=False` if you don’t need it for a quick pass.

---

## Inference

Use the tiny helper to caption a single local image with any BLIP checkpoint (pretrained or your fine‑tuned export):

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from blip_caption.inference import caption_image

# Load either your trained artifacts or a pretrained BLIP:
processor = BlipProcessor.from_pretrained("blip-open-out")  # or "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained("blip-open-out")

print(caption_image("path/to/image.jpg", processor, model, max_new_tokens=30, num_beams=5))
```

You can also point both `from_pretrained` calls to `"Salesforce/blip-image-captioning-base"` for quick zero‑shot captions.

---

## Design notes (what’s under the hood)

- **Data** (`data.py`)
  - Flickr8k: loaded from HF Datasets, split into 90/5/5 with a seed for determinism. We cast `image` to a deferred `Image` feature to inject a stable `image_id` from filenames, then decode on demand.
  - COCO‑Karpathy: uses the community HF dataset. We keep the standard columns (`image`, `caption`, `image_id`) and normalize the records (when captions are lists, we sample one per example for supervision).

- **Collation** (`collator.py`)
  - Uses `BlipProcessor` to tokenize text and process images in one shot, returns `pixel_values`, `input_ids`, `attention_mask`, and `labels` where pad tokens are masked to `-100`.

- **Modeling** (`modeling.py`)
  - BLIP base (ViT‑B/16 backbone) via `BlipForConditionalGeneration`. Optional “freeze vision” switch for parameter‑efficient fine‑tuning.

- **Training** (`training.py`)
  - Standard `TrainingArguments` + `Trainer` with BLEU compute function (via `evaluate`/`sacrebleu`). Logging to TensorBoard out of the box. Checkpoint churn is capped.

- **Metrics** (`metrics.py`)
  - Post‑training evaluation computes COCO‑style metrics. SPICE is behind a toggle and sub‑samples to keep runtime reasonable in local setups.

---

## Reproducibility

- `seed_utils.set_seed` seeds Python, NumPy, and PyTorch RNGs.
- When using CUDA, you’ll get stable results *per hardware/driver combo*; minor variance across GPUs is normal in practice.
- For bit‑for‑bit determinism, also set `CUBLAS_WORKSPACE_CONFIG=:16:8` or `:4096:8` and disable certain CuDNN autotune knobs (not wired here by default).

---

## GPU & memory guidance

- **BLIP base** comfortably trains on a **single 16GB GPU** with the defaults (gradient checkpointing helps). If you’re short on memory:
  - Set `freeze_vision=True`.
  - Lower `per_device_train_bs` and/or increase `grad_accum_steps`.
  - Consider enabling 8‑bit Adam (`bitsandbytes`) or LoRA if you need more headroom (not included here).

---

## Troubleshooting

**COCO/SPICE errors (Java required)**  
- Install Java 8+ (OpenJDK is fine). On Ubuntu:
  ```bash
  sudo apt-get update && sudo apt-get install -y default-jre
  java -version  # should print something like "1.8.x" or "11.x"
  ```
- Re‑run; the first SPICE evaluation downloads Stanford CoreNLP assets into a cache folder.  
- If you don’t need SPICE, set `coco.do_spice=False` in `config.py`.

**PyTorch + CUDA**  
- Always install the **CUDA‑matched** wheel from the PyTorch download page. A mismatched runtime is the #1 source of perf/installation issues.

**HF Datasets timeouts**  
- The first run will cache data in `~/.cache/huggingface/datasets`. Use `HF_DATASETS_CACHE=/custom/path` if you need to relocate it.

---

## Extending this repo (suggested next steps)

- Add a small CLI (Typer/Hydra) to override any config field without editing the file.
- Swap models: try `Salesforce/blip-image-captioning-large` or BLIP‑2 for stronger zero‑shot.
- Integrate parameter‑efficient fine‑tuning (LoRA/IA³) for smaller GPUs.
- Export to ONNX/OpenVINO for CPU‑only inference; add a minimal FastAPI/Gradio demo.
- Add unit tests for the collator and metric wrappers; wire GitHub Actions for lint/test.

---

## Acknowledgments & references

- **BLIP paper**: *Bootstrapping Language‑Image Pre‑training for Unified Vision‑Language Understanding and Generation*, Li et al., 2022.  
- **Salesforce BLIP model cards** (Hugging Face).  
- **Official BLIP code** (Salesforce Research).  
- **LAVIS**: a general library for language‑vision models built by Salesforce Research.

---

## Citation

If you use this codebase in academic work, please cite BLIP:

```bibtex
@misc{li2022blip,
  title  = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  year   = {2022},
  eprint = {2201.12086},
  archivePrefix = {arXiv}
}
```

---

## License

This repository’s code is provided as‑is under the project’s root license (if not specified, treat it as “all rights reserved”). The BLIP weights and datasets have their **own licenses/terms**—please review them before training or shipping a product.

