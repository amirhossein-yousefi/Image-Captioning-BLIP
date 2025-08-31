from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

from blip_caption.config import AllConfig

def build_processor(cfg: AllConfig) -> BlipProcessor:
    return BlipProcessor.from_pretrained(cfg.paths.model_id)

def build_model(cfg: AllConfig) -> BlipForConditionalGeneration:
    model = BlipForConditionalGeneration.from_pretrained(cfg.paths.model_id)
    # Make Trainer happy: give integer max_length with headroom
    model.generation_config.max_length = cfg.gen.total_max_len
    model.generation_config.num_beams = cfg.gen.num_beams
    model.generation_config.length_penalty = cfg.gen.length_penalty
    model.generation_config.early_stopping = cfg.gen.early_stopping
    # Clear max_new_tokens avoid confusion (we pass max_length)
    model.generation_config.max_new_tokens = None

    if cfg.train.freeze_vision and hasattr(model, "vision_model"):
        for p in model.vision_model.parameters():
            p.requires_grad = False
        print("[Info] Vision encoder frozen.")

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model

def use_fp16() -> bool:
    return torch.cuda.is_available()
