from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class PathsConfig:
    model_id: str = "Salesforce/blip-image-captioning-base"
    output_dir: str = "blip-open-out"

@dataclass
class DataConfig:
    # "flickr8k" (default) | "coco_karpathy"
    use_dataset: str = "flickr8k"
    flickr8k_hf_id: str = "ariG23498/flickr8k"
    coco_karpathy_hf_id: str = "yerevann/coco-karpathy"
    seed: int = 42

@dataclass
class TrainConfig:
    epochs: int = 4
    lr: float = 5e-5
    per_device_train_bs: int = 8
    per_device_eval_bs: int = 8
    grad_accum_steps: int = 2
    gradient_checkpointing: bool = True
    freeze_vision: bool = False
    logging_steps: int = 50
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "sacrebleu"
    greater_is_better: bool = True
    # keep as tuple (immutable). training.py converts to list for HF args
    report_to: Tuple[str, ...] = ("tensorboard",)

@dataclass
class GenerationConfig:
    max_txt_len: int = 40
    gen_max_new_tokens: int = 30
    num_beams: int = 5
    length_penalty: float = 1.0
    early_stopping: bool = True

    @property
    def total_max_len(self) -> int:
        return self.max_txt_len + self.gen_max_new_tokens

@dataclass
class COCOMetricConfig:
    do_spice: bool = True
    spice_subsample: int | None = 1500
    save_coco_metrics_json: bool = True

@dataclass
class AllConfig:
    # Use default_factory to avoid shared mutable defaults
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    coco: COCOMetricConfig = field(default_factory=COCOMetricConfig)
