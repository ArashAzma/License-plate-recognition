from transformers import AutoModel, AutoTokenizer
from preprocess_image import save_temp_image
import os

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

def apply_ocr(image, ocr_type='ocr', **kwargs):
    image_path = save_temp_image(image)
    try:
        result = model.chat(tokenizer, image_path, ocr_type=ocr_type, **kwargs)
    finally:
        os.remove(image_path)
    
    return result