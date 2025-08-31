import io, json, base64, logging, os
from typing import Any, Dict, Tuple, List, Optional
from urllib.parse import urlparse

import torch
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Predictor:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, images: List[Image.Image], prompt: Optional[str], gen_kwargs: Dict[str, Any]):
        inputs = self.processor(images=images, text=[prompt]*len(images) if prompt else None,
                                return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, **gen_kwargs)
        texts = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [t.strip() for t in texts]

def _load_image_from_json(obj: Any) -> Image.Image:
    if isinstance(obj, dict):
        if "b64" in obj:
            return Image.open(io.BytesIO(base64.b64decode(obj["b64"]))).convert("RGB")
        if "url" in obj:
            url = obj["url"]
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    raise ValueError("Provide image as {'b64': ...} or {'url': ...}")

# ---- SageMaker hooks ----
def model_fn(model_dir: str) -> Predictor:
    LOGGER.info(f"Loading model from: {model_dir}")
    return Predictor(model_dir)

def input_fn(request_body: bytes, request_content_type: str) -> Tuple[List[Image.Image], Dict[str, Any]]:
    if request_content_type != "application/json":
        raise ValueError("Only application/json supported.")
    payload = json.loads(request_body.decode("utf-8"))

    instances = payload.get("instances") or payload.get("inputs")
    if not instances:
        raise ValueError("Missing 'instances' or 'inputs'.")

    images = [_load_image_from_json(inst) for inst in instances]
    parameters = payload.get("parameters", {})
    prompt = payload.get("prompt")
    # reasonable defaults for captioning
    gen_kwargs = {
        "max_new_tokens": int(parameters.get("max_new_tokens", 30)),
        "num_beams": int(parameters.get("num_beams", 3)),
        "do_sample": bool(parameters.get("do_sample", False)),
        "temperature": float(parameters.get("temperature", 1.0)),
        "top_p": float(parameters.get("top_p", 1.0)),
    }
    return images, {"prompt": prompt, "gen_kwargs": gen_kwargs}

def predict_fn(data: Tuple[List[Image.Image], Dict[str, Any]], model: Predictor) -> Dict[str, Any]:
    images, opts = data
    caps = model.generate(images, opts["prompt"], opts["gen_kwargs"])
    return {"predictions": [{"generated_text": t} for t in caps]}

def output_fn(prediction: Dict[str, Any], accept: str) -> bytes:
    if accept not in ("application/json", "*/*"):
        raise ValueError("Only application/json supported.")
    return json.dumps(prediction).encode("utf-8")
