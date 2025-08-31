from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import evaluate

def build_bleu_fn(tokenizer):
    """
    Returns a compute_metrics function for HF Trainer that computes SacreBLEU.
    Uses the provided tokenizer for decoding preds/labels.
    """
    sacrebleu = evaluate.load("sacrebleu")

    def compute_bleu(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        dec_preds = [p.strip() for p in dec_preds]
        dec_labels = [[l.strip()] for l in dec_labels]
        bleu = sacrebleu.compute(predictions=dec_preds, references=dec_labels)["score"]
        return {"sacrebleu": float(bleu)}

    return compute_bleu

# ---------------- COCO-style metrics ----------------

def compute_coco_metrics(
    decoded_preds: List[str],
    val_ds,
    do_spice: bool = True,
    spice_subsample: Optional[int] = 1500,
) -> Dict[str, Optional[float]]:
    """
    Build per-image maps (1 pred/image, N refs/image) and compute COCO metrics.
    """
    # Build refs: all captions per image_id in val split
    refs_map: Dict[str, List[str]] = {}
    for iid, cap in zip(val_ds["image_id"], val_ds["caption"]):
        refs_map.setdefault(iid, []).append(cap)

    # Use first prediction per image_id
    pred_map: Dict[str, str] = {}
    for iid, pred in zip(val_ds["image_id"], decoded_preds):
        if iid not in pred_map:
            pred_map[iid] = pred.strip()

    # Align keys; subsample for SPICE if configured
    common_ids = list(set(refs_map) & set(pred_map))
    if spice_subsample and do_spice and len(common_ids) > spice_subsample:
        import random
        common_ids = random.sample(common_ids, spice_subsample)

    # Build COCO-eval input format
    res = {iid: [{"caption": pred_map[iid]}] for iid in common_ids}
    gts = {iid: [{"caption": c} for c in refs_map[iid]] for iid in common_ids}

    # Tokenize
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    metrics: Dict[str, Optional[float]] = {}

    # BLEU (report BLEU-4)
    from pycocoevalcap.bleu.bleu import Bleu
    b_scorer = Bleu(4)
    b_score, _ = b_scorer.compute_score(gts_tok, res_tok)  # list of 4 scores
    metrics["Bleu_4"] = float(b_score[3])

    # METEOR
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        m_scorer = Meteor()
        m_score, _ = m_scorer.compute_score(gts_tok, res_tok)
        metrics["METEOR"] = float(m_score)
    except Exception as e:
        print("[Warn] METEOR failed (needs Java):", e)
        metrics["METEOR"] = None

    # CIDEr
    try:
        from pycocoevalcap.cider.cider import Cider
        c_scorer = Cider()
        c_score, _ = c_scorer.compute_score(gts_tok, res_tok)
        metrics["CIDEr"] = float(c_score)
    except Exception as e:
        print("[Warn] CIDEr failed:", e)
        metrics["CIDEr"] = None

    # SPICE
    if do_spice:
        try:
            from pycocoevalcap.spice.spice import Spice
            s_scorer = Spice()
            s_score, _ = s_scorer.compute_score(gts_tok, res_tok)  # avg F-score
            metrics["SPICE"] = float(s_score)
        except Exception as e:
            print("[Warn] SPICE failed (needs Java & downloads CoreNLP on first run):", e)
            metrics["SPICE"] = None

    return metrics
