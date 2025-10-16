import cv2, time, torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "apple/FastVLM-0.5B"
MAX_SIDE = 640  
MAX_NEW_TOKENS = 48

def resize_keep_aspect(bgr, max_side):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side: return bgr
    s = max_side / float(max(h, w))
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

# --- Load model on GPU if available ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, trust_remote_code=True)
if torch.cuda.is_available():
    model = model.to("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

# --- Sanity prints ---
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)

# --- Open camera ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS,          60)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# --- Measure end-to-end FPS over a sliding window ---
t_last = time.time()
frames = 0

def caption(pil_img):
    # Build a minimal chat-turn with an <image> placeholder
    messages = [{"role": "user", "content": "<image>\nDescribe this image."}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)
    pre_ids  = tokenizer(pre,  return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids

    IMAGE_TOKEN_INDEX = -200
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device, non_blocking=True)
    attention = torch.ones_like(input_ids, device=device)

    vision = model.get_vision_tower()
    px = vision.image_processor(images=pil_img, return_tensors="pt")["pixel_values"]
    px = px.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)

    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        ids = model.generate(inputs=input_ids, attention_mask=attention, images=px,
                             max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if device.type == "cuda": torch.cuda.synchronize()
    print(f"Model latency: {(time.time()-t0)*1000:.0f} ms")
    return tokenizer.decode(ids[0], skip_special_tokens=True)

last_caption = ""
caption_change_threshold = 0.5  # Adjust this value (0.0 to 1.0)

print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok: continue

    frames += 1
    # Check if a frame should be processed
    if frames % 5 != 0:
        cv2.imshow("Live", frame)
    else:
        small = resize_keep_aspect(frame, MAX_SIDE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        
        # Generate a new caption
        new_caption = caption(pil)
        
        # Compare new caption with the last one
        if not last_caption:
            # First run, just set the caption
            last_caption = new_caption
            print("Caption:", last_caption)
        else:
            # Calculate the similarity between captions
            # Using Jaccard similarity as a simple metric
            set1 = set(last_caption.lower().split())
            set2 = set(new_caption.lower().split())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jaccard_similarity = intersection / union if union > 0 else 0

            if jaccard_similarity < caption_change_threshold:
                # If captions are sufficiently different, update
                last_caption = new_caption
                print("Caption:", last_caption)
        
        # Display the frame with the current (potentially old) caption
        cv2.putText(small, last_caption[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Live", small)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # simple FPS display
    now = time.time()
    if now - t_last >= 2.0:
        print(f"E2E FPS ~ {frames/(now - t_last):.1f}")
        frames, t_last = 0, now

cap.release()
cv2.destroyAllWindows()