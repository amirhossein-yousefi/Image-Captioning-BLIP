from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def caption_image(
    path: str,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    max_new_tokens: int = 30,
    num_beams: int = 5,
) -> str:
    image = Image.open(path).convert("RGB")
    enc = processor(images=image, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return processor.tokenizer.decode(out[0], skip_special_tokens=True)
