# Fine-tune BLIP for image captioning on SageMaker.
import os, json, logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Image as HFImage
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BlipProcessor,
    BlipForConditionalGeneration,
    set_seed,
)
import evaluate

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------- Args ---------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Salesforce/blip-image-captioning-base",
        metadata={"help": "HF model id or local path"},
    )

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HF dataset name (optional)"})
    train_file: Optional[str] = field(default=None, metadata={"help": "train.jsonl path (relative ok)"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "validation.jsonl path (relative ok)"})
    image_root: Optional[str] = field(default=None, metadata={"help": "folder containing images (relative ok)"})
    image_column: str = field(default="image")
    caption_column: str = field(default="text")
    max_length: int = field(default=64)

@dataclass
class TrainArguments(Seq2SeqTrainingArguments):
    # Defaults tuned for a single g5.2xlarge; adjust per instance/memory.
    output_dir: str = field(
        default_factory=lambda: os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    fp16: bool = True
    predict_with_generate: bool = True
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    report_to: List[str] = field(default_factory=lambda: ["none"])  # CloudWatch picks logs by default

# --------- Helpers ---------
def _resolve_channel_path(channel_env: str, provided: Optional[str]) -> Optional[str]:
    """Join relative user-provided path with SM channel directory if needed."""
    if not provided:
        return None
    if os.path.isabs(provided):
        return provided
    base = os.environ.get(channel_env)
    return os.path.join(base, provided) if base else provided

def _load_jsonl_dataset(train_file, val_file, image_root, image_col, text_col) -> DatasetDict:
    files = {}
    if train_file: files["train"] = train_file
    if val_file: files["validation"] = val_file
    ds = load_dataset("json", data_files=files)

    # Normalize columns to ["image", "text"]
    if image_col != "image":
        ds = ds.rename_column(image_col, "image")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    # Join image paths with image_root if relative
    if image_root:
        def join_path(example):
            p = example["image"]
            if isinstance(p, str) and not os.path.isabs(p):
                example["image"] = os.path.join(image_root, p)
            return example
        ds = ds.map(join_path)

    ds = ds.cast_column("image", HFImage(decode=True))
    return ds

class BlipDataCollator:
    """Build pixel_values + labels on the fly to keep dataset light & flexible."""
    def __init__(self, processor: BlipProcessor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [f["image"] for f in features]
        texts = [f["text"] for f in features]
        enc = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100  # ignore index
        return {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

# --------- Main ---------
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # Resolve channel-bound paths (SageMaker downloads S3 data here)
    # /opt/ml/input/data/train, /opt/ml/input/data/validation
    train_file = _resolve_channel_path("SM_CHANNEL_TRAIN", data_args.train_file)
    val_file = _resolve_channel_path("SM_CHANNEL_VALIDATION", data_args.validation_file)
    image_root = _resolve_channel_path("SM_CHANNEL_TRAIN", data_args.image_root or "")

    LOGGER.info(f"Train file: {train_file}, Val file: {val_file}, Image root: {image_root}")

    set_seed(train_args.seed)

    processor = BlipProcessor.from_pretrained(model_args.model_name_or_path)
    model = BlipForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    # Load data
    if data_args.dataset_name:
        ds = load_dataset(data_args.dataset_name)
        # best-effort normalization
        if "caption" in ds["train"].column_names and "text" not in ds["train"].column_names:
            ds = ds.rename_column("caption", "text")
        if "image" in ds["train"].column_names:
            ds = ds.cast_column("image", HFImage(decode=True))
    else:
        ds = _load_jsonl_dataset(train_file, val_file, image_root, data_args.image_column, data_args.caption_column)

    # Metric: SacreBLEU on generated captions
    sacrebleu = evaluate.load("sacrebleu")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]
        return sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)

    collator = BlipDataCollator(processor, data_args.max_length)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=ds.get("train"),
        eval_dataset=ds.get("validation"),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(train_args.output_dir)
    processor.save_pretrained(train_args.output_dir)

    # small hint for the default HF inference toolkit
    with open(os.path.join(train_args.output_dir, "task.json"), "w") as f:
        json.dump({"task": "image-to-text"}, f)

if __name__ == "__main__":
    main()
