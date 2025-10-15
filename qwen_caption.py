# qwen2vl_caption_ar.py
import cv2, torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    messages = [{"role":"user","content":[{"type":"image","image":img},
               {"type":"text","text":"صف هذه الصورة بالعربية بجملة قصيرة وبسيطة."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(model.device)

    out_ids = model.generate(**inputs, max_new_tokens=64)
    gen_ids_trim = [o[len(i):] for i,o in zip(inputs.input_ids, out_ids)]
    caption_ar = processor.batch_decode(gen_ids_trim, skip_special_tokens=True)[0]

    cv2.putText(frame, caption_ar, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Qwen2-VL (Arabic)", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
# pip install "transformers>=4.43" qwen-vl-utils pillow accelerate torch --upgrade
