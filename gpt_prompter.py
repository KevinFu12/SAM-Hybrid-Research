import os
import random
import base64
import io
import numpy as np
from PIL import Image
from openai import OpenAI
from config import MEDICAL_PROMPTS

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def encode_numpy_image_to_base64(image_np):
    """Convert numpy image to base64 string"""
    img_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ask_gpt_prompt(image_np):
    """Get wound description from GPT-4o"""
    base64_image = encode_numpy_image_to_base64(image_np)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "You are a medical imaging expert. Describe any visible wounds on this diabetic foot. "
                "Include location, severity, color, and shape. Respond with one sentence.")},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    }]
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❗ GPT-4o Error:", e)
        return random.choice(MEDICAL_PROMPTS)

def is_plural_wound(prompt):
    """Check if prompt indicates multiple wounds"""
    plural_indicators = ["multiple", "several", "many", "wounds", "ulcers", "lesions"]
    return any(word in prompt.lower() for word in plural_indicators)

def validate_gpt_response(response):
    """Validate and potentially replace GPT response"""
    invalid_indicators = ["sorry", "can't", "unsure", "unable", "no visible", "cannot"]
    if any(x in response.lower() for x in invalid_indicators):
        return random.choice(MEDICAL_PROMPTS)
    return response