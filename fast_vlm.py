import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration / model loading ---

MODEL_ID = "apple/FastVLM-0.5B"  # اسم النموذج على Hugging Face أو ما تحمّله محليًا

def load_model(model_id):
    # Load tokenizer (with trust_remote_code=True إذا النموذج يعدل أو يستخدم كود مخصص)  
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    # Load model، نستخدم torch_dtype=torch.float16 إذا GPU يدعم، ونرسله إلى CUDA إذا متاح
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        # يمكنك استخدام device_map="auto" أو manual placement
        device_map="auto"
    )
    return tokenizer, model

# Function to generate caption from an image
def caption_image(tokenizer, model, pil_img, max_new_tokens=50):
    # Prepare prompt / template مع وسم الصورة
    messages = [
        {"role": "user", "content": "<image>\nDescribe this image in detail in arabic."}
    ]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # Split حول وسم <image>
    pre, post = rendered.split("<image>", 1)
    # ترميز النصوص قبل وبعد
    pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
    # IMAGE_TOKEN_INDEX هو مؤشر يُستخدم كتمثيل للصورة في تسلسل الإدخال
    IMAGE_TOKEN_INDEX = -200  # النموذج قد يستخدم قيمة ثابتة كهذه
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    # معالجة الصورة: النموذج يوفر وظيفة get_vision_tower
    px = model.get_vision_tower().image_processor(images=pil_img, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=model.dtype)
    # توليد النصوص (بدون حساب التدرجات)
    with torch.no_grad():
        output_ids = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=px,
            max_new_tokens=max_new_tokens,
        )
    # فك ترميز المخرجات إلى نص
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def main():
    # تحميل النموذج
    tokenizer, model = load_model(MODEL_ID)
    # فتح الكاميرا (0 = الكاميرا الافتراضية)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    print("Starting live captioning. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV يعطي BGR، نحوله إلى RGB و PIL Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # توليد التسمية الوصفية للصورة
        caption = caption_image(tokenizer, model, pil_img, max_new_tokens=40)

        # طباعة الوصف في سطر الأوامر
        print("Caption:", caption)

        # عرض الإطار مع العنوان (اختياري)
        cv2.putText(frame, caption, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Live Camera", frame)

        # الانتظار لمفتاح “q” للخروج
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
