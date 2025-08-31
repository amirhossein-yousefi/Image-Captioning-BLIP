import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from blip_caption.config import AllConfig
from blip_caption.modeling import use_fp16

def build_training_args(cfg: AllConfig) -> Seq2SeqTrainingArguments:
    fp16 = use_fp16()
    return Seq2SeqTrainingArguments(
        output_dir=cfg.paths.output_dir,
        num_train_epochs=cfg.train.epochs,
        learning_rate=cfg.train.lr,
        per_device_train_batch_size=cfg.train.per_device_train_bs,
        per_device_eval_batch_size=cfg.train.per_device_eval_bs,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=cfg.train.logging_steps,
        predict_with_generate=True,
        generation_max_length=cfg.gen.total_max_len,
        generation_num_beams=cfg.gen.num_beams,
        remove_unused_columns=False,
        report_to=list(cfg.train.report_to),
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        save_total_limit=cfg.train.save_total_limit,
        logging_dir=os.path.join(cfg.paths.output_dir, "logs"),
    )

def build_trainer(
    model,
    processor,
    datasets,
    args: Seq2SeqTrainingArguments,
    collator,
    compute_metrics,
) -> Seq2SeqTrainer:
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer
