# pip install -U "transformers>=4.43" qwen-vl-utils accelerate pillow opencv-python
import os, time, threading, queue, cv2, torch, numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ----------------- Perf setup -----------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ----------------- Processor/Model -----------------
# Fewer visual tokens => much faster
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=224*224,
    max_pixels=28*28*128   # if still slow, try 96 or 64
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=dtype, device_map={"": 0} if use_cuda else {"": "cpu"}
).eval()

# ----------------- Caption helpers -----------------
def make_prompt(img):
    msgs = [{"role":"user","content":[
        {"type":"image","image":img},
        {"type":"text","text":"Describe this image in one short, simple English sentence."}
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    return text, image_inputs, video_inputs

def caption_once(frame_bgr):
    # downscale further before sending to model (extra speed)
    fh, fw = frame_bgr.shape[:2]
    scale = 256 / max(fh, fw)
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(fw*scale), int(fh*scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    text, image_inputs, video_inputs = make_prompt(img)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    if use_cuda:
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)  # short & greedy
    gen = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    return processor.batch_decode(gen, skip_special_tokens=True)[0]

# ----------------- Caption rendering (wrapped + background) -----------------
def _wrap_lines(text, font, scale, thickness, max_width_px):
    words = (text or "").split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        (w_px, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if w_px <= max_width_px or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def render_caption(frame_bgr, text, where="bottom",
                   font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, thickness=2,
                   fg=(0,255,0), bg=(0,0,0), alpha=0.35,
                   margin=8, line_gap=6, max_width_ratio=0.92):
    if not text: return frame_bgr
    h, w = frame_bgr.shape[:2]
    max_width_px = int(w * max_width_ratio)

    lines = _wrap_lines(text, font, scale, thickness, max_width_px)
    sizes = [cv2.getTextSize(l, font, scale, thickness)[0] for l in lines] or [(0,0)]
    line_h = max(s[1] for s in sizes) or 0
    block_w = min(max((s[0] for s in sizes), default=0), max_width_px)
    block_h = len(lines)*line_h + (len(lines)-1)*line_gap + 2*margin

    x0 = margin
    y0 = (h - block_h - margin) if where.lower() == "bottom" else margin

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+block_w+2*margin, y0+block_h), bg, -1)
    frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1-alpha, 0)

    y = y0 + margin + line_h
    for l in lines:
        cv2.putText(frame_bgr, l, (x0 + margin, y), font, scale, fg, thickness, cv2.LINE_AA)
        y += line_h + line_gap
    return frame_bgr

# ----------------- Threads: camera (producer) + caption worker -----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

latest = {"frame": None}
lock   = threading.Lock()
jobs   = queue.Queue(maxsize=1)
result = {"text": ""}

def grabber():
    while True:
        ok, f = cap.read()
        if not ok: break
        with lock:
            latest["frame"] = f

def worker():
    while True:
        _ = jobs.get()
        with lock:
            f = None if latest["frame"] is None else latest["frame"].copy()
        if f is None:
            result["text"] = ""
            continue
        try:
            result["text"] = caption_once(f)
        except Exception:
            result["text"] = result.get("text", "")

threading.Thread(target=grabber, daemon=True).start()
threading.Thread(target=worker,  daemon=True).start()

# Warm-up once
time.sleep(0.3)
with lock:
    if latest["frame"] is not None and jobs.empty():
        jobs.put_nowait(None)

INTERVAL = 2.0   # caption every 2s (press Space to force)
last_t = 0.0

# ----------------- Main UI loop -----------------
while True:
    with lock:
        frame = None if latest["frame"] is None else latest["frame"].copy()
    if frame is None:
        if cv2.waitKey(1) & 0xFF == 27: break
        continue

    now = time.time()
    key = cv2.waitKey(1) & 0xFF
    if (now - last_t > INTERVAL) or (key == 32):  # Space triggers immediate caption
        if jobs.empty():
            jobs.put_nowait(None)
            last_t = now

    frame = render_caption(frame, result["text"], where="bottom")  # or where="top"
    cv2.imshow("Qwen2-VL English (FAST mode)", frame)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
