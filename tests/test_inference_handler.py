import base64, json
from PIL import Image
import io
from sagemaker.inference import input_fn,output_fn,predict_fn

def _fake_image_b64():
    im = Image.new("RGB", (16, 16), color=(128, 200, 255))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def test_input_output_roundtrip(monkeypatch):
    # Skip model loading by faking Predictor
    class FakePredictor:
        def generate(self, images, prompt, gen_kwargs):
            return ["a small blue square"] * len(images)
    pred = FakePredictor()

    payload = {"instances": [{"b64": _fake_image_b64()}], "parameters": {"max_new_tokens": 5}}
    data = input_fn(json.dumps(payload).encode(), "application/json")
    out = predict_fn(data, pred)
    body = output_fn(out, "application/json")
    parsed = json.loads(body.decode())
    assert "predictions" in parsed and len(parsed["predictions"]) == 1
