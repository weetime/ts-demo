# image_recognition.py
import warnings
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# 忽略 FutureWarning
warnings.simplefilter("ignore", FutureWarning)

# 加载预训练模型
model_name = "google/vit-base-patch16-224"  # 这里你可以选择任何预训练的模型
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def recognize_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    print(recognize_image(image_path))
