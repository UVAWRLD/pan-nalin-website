# Segment-Anything masks from point prompts.
#
#   python tools/scene3d/segment.py <image> <prompts.json> <out_dir>
#
# prompts.json: { "name": { "pos": [[x, y], ...], "neg": [[x, y], ...], "box": [x0, y0, x1, y1]? }, ... }
# with coordinates in a 1920-wide version of the image (the size it is viewed at).
# Writes <out_dir>/<name>.png (0/255) at full image size.
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

src, prompts_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
image = Image.open(src).convert("RGB")
k = image.width / 1920.0
prompts = json.load(open(prompts_path))

processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    emb = model.get_image_embeddings(inputs["pixel_values"].to(device, torch.float32))

for name, p in prompts.items():
    pts = [[x * k, y * k] for x, y in p.get("pos", []) + p.get("neg", [])]
    labels = [1] * len(p.get("pos", [])) + [0] * len(p.get("neg", []))
    kw = {}
    if pts:
        kw["input_points"] = [[pts]]
        kw["input_labels"] = [[labels]]
    if "box" in p:
        kw["input_boxes"] = [[[v * k for v in p["box"]]]]
    inp = processor(images=image, return_tensors="pt", **kw)
    args = {"image_embeddings": emb, "multimask_output": True}
    for key in ("input_points", "input_labels", "input_boxes"):
        if key in inp:
            v = inp[key]
            args[key] = v.to(device, torch.float32 if key != "input_labels" else torch.int64)
    with torch.no_grad():
        out = model(**args)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0]
    scores = out.iou_scores.cpu()[0, 0]
    pick = p.get("pick", int(scores.argmax()))
    m = masks[pick].numpy().astype(np.uint8) * 255
    Image.fromarray(m).save(out_dir / f"{name}.png")
    print(name, "scores", [round(float(s), 3) for s in scores], "pick", pick, "area", int((m > 0).sum()))
