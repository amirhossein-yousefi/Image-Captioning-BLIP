from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import BlipProcessor

from .config import AllConfig

@dataclass
class ImageCaptionCollator:
    processor: BlipProcessor
    max_length: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [ex["image"] for ex in batch]   # PIL images
        texts  = [ex["caption"] for ex in batch]

        enc = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Labels from input_ids with pad masked to -100
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        return {
            "pixel_values":   enc["pixel_values"],
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
        }

def build_collator(processor: BlipProcessor, cfg: AllConfig) -> ImageCaptionCollator:
    return ImageCaptionCollator(processor=processor, max_length=cfg.gen.max_txt_len)
