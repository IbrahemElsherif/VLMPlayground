# pip install -U "transformers>=4.43" qwen-vl-utils accelerate pillow opencv-python arabic-reshaper python-bidi sentencepiece sacremoses
import os, time, threading, queue, cv2, torch, numpy as np, os.path as osp
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from qwen_vl_utils import process_vision_info
import arabic_reshaper
from bidi.algorithm import get_display

# ----------------- Settings -----------------
MIRROR = True                  # True = un-mirror selfie webcams so Arabic isn't flipped
INTERVAL = 2.0                 # caption every N seconds (Space = force now)
CAM_W, CAM_H = 320, 240        # capture size (smaller -> faster)
VIS_MAX_TOKENS = 28*28*128     # try 96 or 64 if still slow
MAX_NEW_TOKENS = 10            # shorter = faster

# Choose translator: "marian" (fast, small) or "nllb" (bigger, higher quality)
TRANSLATOR = "marian"          # change to "nllb" to use NLLB-200

# ----------------- Perf / HF Hub setup -----------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
torch.backends.cuda.matmul.allow_tf32 = True
use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ----------------- Processor/Model (avoid meta tensors) -----------------
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=224*224,
    max_pixels=VIS_MAX_TOKENS
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=(torch.float16 if use_cuda else torch.float32),
    device_map=("auto" if use_cuda else None),
    low_cpu_mem_usage=False
).eval()

# ----------------- Translation models -----------------
if TRANSLATOR.lower() == "marian":
    # Fast & lightweight EN->AR
    tr_id = "Helsinki-NLP/opus-mt-en-ar"
    tr_tok = AutoTokenizer.from_pretrained(tr_id)
    tr_mt  = AutoModelForSeq2SeqLM.from_pretrained(tr_id, use_safetensors=True).to("cpu").eval()

    def en2ar(text: str) -> str:
        try:
            inputs = tr_tok(text, return_tensors="pt")
            out = tr_mt.generate(**inputs, max_new_tokens=96, num_beams=1, do_sample=False)
            return tr_tok.batch_decode(out, skip_special_tokens=True)[0]
        except Exception:
            return text

elif TRANSLATOR.lower() == "nllb":
    # Higher quality, larger model; supports many dialect codes
    tr_id = "facebook/nllb-200-distilled-600M"
    tr_tok = AutoTokenizer.from_pretrained(tr_id)
    tr_mt  = AutoModelForSeq2SeqLM.from_pretrained(tr_id, use_safetensors=True).to("cpu").eval()
    SRC_CODE = "eng_Latn"
    TGT_CODE = "arb_Arab"  # Modern Standard Arabic; try "ary_Arab" (Moroccan), "apc_Arab" (Levantine), etc.

    def en2ar(text: str) -> str:
        try:
            tr_tok.src_lang = SRC_CODE
            inputs = tr_tok(text, return_tensors="pt")
            out = tr_mt.generate(
                **inputs,
                forced_bos_token_id=tr_tok.lang_code_to_id[TGT_CODE],
                max_new_tokens=96,
                num_beams=1,
                do_sample=False
            )
            return tr_tok.batch_decode(out, skip_special_tokens=True)[0]
        except Exception:
            return text
else:
    raise ValueError("TRANSLATOR must be 'marian' or 'nllb'.")

# ----------------- Font (Arabic) -----------------
def load_arabic_font(size=28):
    candidates = [
        r"NotoNaskhArabic-Regular.ttf",
        r"C:\Windows\Fonts\NotoNaskhArabic-Regular.ttf",
        r"C:\Windows\Fonts\NotoSansArabic-Regular.ttf",
        r"C:\Windows\Fonts\Tahoma.ttf",
    ]
    for p in candidates:
        if osp.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    raise FileNotFoundError(
        "Place an Arabic TTF (e.g., NotoNaskhArabic-Regular.ttf) and update path above."
    )
font = load_arabic_font(28)

# ----------------- Caption helpers -----------------
def make_prompt(img: Image.Image):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image in one short, simple English sentence."}
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    return text, image_inputs, video_inputs

def caption_once(frame_bgr):
    # extra downscale before sending to model
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
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    caption_en = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return en2ar(caption_en)  # Arabic translation

# ----------------- Cached Arabic overlay renderer -----------------
_render_cache = {"text": None, "overlay": None}

def render_caption_overlay_rgba(frame_size, text, where="bottom",
                                fg=(0,255,0), bg=(0,0,0), alpha=0.35,
                                margin=8, line_gap=6, max_width_ratio=0.92):
    """
    Returns an RGBA overlay (PIL.Image) same size as frame; re-renders only when 'text' changes.
    """
    W, H = frame_size
    if not text:
        return Image.new("RGBA", (W, H), (0,0,0,0))

    if _render_cache["text"] == text and _render_cache["overlay"] is not None:
        return _render_cache["overlay"]

    txt = get_display(arabic_reshaper.reshape(text))
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    max_w = int(W * max_width_ratio)

    # wrap lines (measure with PIL)
    words, lines, cur = txt.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        width = int(draw.textlength(test, font=font))
        if width <= max_w or not cur:
            cur = test
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)

    # block metrics
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    block_w = min(max(int(draw.textlength(l, font=font)) for l in lines), max_w) if lines else 0
    block_h = len(lines)*line_h + (len(lines)-1)*line_gap + 2*margin

    # RIGHT anchor for Arabic
    x0 = W - (block_w + 2*margin) - margin
    y0 = (H - block_h - margin) if where.lower() == "bottom" else margin

    # translucent background
    draw.rectangle([x0, y0, x0+block_w+2*margin, y0+block_h], fill=(*bg, int(alpha*255)))

    # draw lines right-aligned
    y = y0 + margin
    for l in lines:
        lw = int(draw.textlength(l, font=font))
        draw.text((x0 + (block_w - lw) + margin, y), l, font=font, fill=(*fg,255))
        y += line_h + line_gap

    _render_cache["text"] = text
    _render_cache["overlay"] = overlay
    return overlay

# ----------------- Threads: camera + caption worker -----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

latest = {"frame": None}
lock   = threading.Lock()
jobs   = queue.Queue(maxsize=1)
result = {"text": ""}

def grabber():
    while True:
        ok, f = cap.read()
        if not ok: break
        if MIRROR:
            f = cv2.flip(f, 1)  # un-mirror the selfie feed so Arabic isn't flipped
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
            result["text"] = result.get("text","")

threading.Thread(target=grabber, daemon=True).start()
threading.Thread(target=worker,  daemon=True).start()

# Warm-up once
time.sleep(0.3)
with lock:
    if latest["frame"] is not None and jobs.empty():
        jobs.put_nowait(None)

last_t = 0.0

# ----------------- Main UI loop -----------------
while True:
    with lock:
        frame = None if latest["frame"] is None else latest["frame"].copy()
        cur_text = result.get("text","")

    if frame is None:
        if cv2.waitKey(1) & 0xFF == 27: break
        continue

    key = cv2.waitKey(1) & 0xFF
    now = time.time()
    if (now - last_t > INTERVAL) or (key == 32):  # Space to force caption
        if jobs.empty():
            jobs.put_nowait(None)
            last_t = now

    # compose cached overlay (re-renders only when cur_text changed)
    H, W = frame.shape[:2]
    try:
        overlay = render_caption_overlay_rgba((W, H), cur_text, where="bottom")
        base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        out = Image.alpha_composite(base, overlay).convert("RGB")
        frame = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    except Exception:
        pass  # show raw frame if drawing fails

    cv2.imshow("Qwen2-VL Arabic (FAST, right-aligned)", frame)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
