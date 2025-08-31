import os
import json
import numpy as np

from blip_caption.config import AllConfig
from blip_caption.seed_utils import set_seed
from blip_caption.data import load_open_dataset
from blip_caption.modeling import build_model, build_processor
from blip_caption.collator import build_collator
from blip_caption.metrics import build_bleu_fn, compute_coco_metrics
from blip_caption.training import build_training_args, build_trainer
from blip_caption.inference import caption_image

def main():
    cfg = AllConfig()

    # ---------- Repro ----------
    set_seed(cfg.data.seed)

    # ---------- Data ----------
    data = load_open_dataset(cfg)

    # ---------- Model & Processor ----------
    processor = build_processor(cfg)
    model = build_model(cfg)

    # ---------- Collator ----------
    collator = build_collator(processor, cfg)

    # ---------- BLEU metric for training-time eval ----------
    compute_bleu = build_bleu_fn(processor.tokenizer)

    # ---------- Training args & Trainer ----------
    args = build_training_args(cfg)
    trainer = build_trainer(
        model=model,
        processor=processor,
        datasets=data,
        args=args,
        collator=collator,
        compute_metrics=compute_bleu,
    )

    # ---------- Train ----------
    train_result = trainer.train()
    print("Best (by BLEU):", trainer.state.best_metric)

    # ---------- COCO-style metrics (CIDEr/METEOR/SPICE) ----------
    pred_out = trainer.predict(data["validation"])
    pred_ids = pred_out.predictions[0] if isinstance(pred_out.predictions, tuple) else pred_out.predictions
    decoded = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    coco_metrics = compute_coco_metrics(
        decoded_preds=decoded,
        val_ds=data["validation"],
        do_spice=cfg.coco.do_spice,
        spice_subsample=cfg.coco.spice_subsample,
    )
    print("COCO-style metrics:", coco_metrics)

    if cfg.coco.save_coco_metrics_json:
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        with open(os.path.join(cfg.paths.output_dir, "coco_metrics.json"), "w") as f:
            json.dump({k: (None if v is None else float(v)) for k, v in coco_metrics.items()}, f, indent=2)

    # ---------- Save model & processor ----------
    trainer.save_model()
    processor.save_pretrained(cfg.paths.output_dir)

    # ---------- Quick inference helper ----------
    print("Tip: Try inference via:")
    print("from blip_caption.inference import caption_image")
    print("caption_image('some_local_image.jpg', processor, model, max_new_tokens=30, num_beams=5)")

if __name__ == "__main__":
    main()
