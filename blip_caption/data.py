import os
import random
from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, Image as HFImage

from .config import AllConfig

def _add_image_id(batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    ids: List[str] = []
    for im in batch["image"]:
        # im may be dict(path=..., bytes=...) when decode=False
        path = im["path"] if isinstance(im, dict) else str(im)
        ids.append(os.path.basename(path))
    return {"image_id": ids}

def _prepare_coco_example(example: Dict[str, Any]) -> Dict[str, Any]:
    s = example["sentences"]
    example["caption"] = random.choice(s) if isinstance(s, list) and len(s) else s
    example["image"] = example["url"]
    example["image_id"] = example.get("filename", str(example.get("cocoid", example.get("imgid", ""))))
    return example

def load_open_dataset(cfg: AllConfig) -> DatasetDict:
    if cfg.data.use_dataset == "flickr8k":
        raw = load_dataset(cfg.data.flickr8k_hf_id)  # single "train" split
        ds = raw["train"]
        tmp = ds.train_test_split(test_size=0.10, seed=cfg.data.seed)  # 90/10
        val_test = tmp["test"].train_test_split(test_size=0.50, seed=cfg.data.seed)  # 5/5

        data = DatasetDict({
            "train": tmp["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        # Stable image_id from filename
        data = data.cast_column("image", HFImage(decode=False))
        data = data.map(_add_image_id, batched=True, desc="Adding image_id")
        data = data.cast_column("image", HFImage(decode=True))
        return data

    elif cfg.data.use_dataset == "coco_karpathy":
        d = load_dataset(cfg.data.coco_karpathy_hf_id)
        data = DatasetDict({
            "train": d["train"],
            "validation": d["validation"],
            "test": d["test"],
        })
        # lazily decode from URL
        data = data.cast_column("url", HFImage())
        keep_cols = {"image", "caption", "image_id"}
        data = data.map(
            _prepare_coco_example,
            desc="Preparing COCO examples",
            remove_columns=[c for c in data["train"].column_names if c not in keep_cols]
        )
        return data

    else:
        raise ValueError(f"Unknown dataset config: {cfg.data.use_dataset}")
