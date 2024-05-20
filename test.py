import warnings
from transformers import pipeline

# 忽略 FutureWarning
warnings.simplefilter("ignore", FutureWarning)

# 加载预训练的情感分析模型，显式指定模型名称和版本
classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

# 使用模型进行推理
results = classifier("I love using Hugging Face's transformers library!")

# 输出结果
for result in results:
    print(f"Label: {result['label']}, with score: {result['score']:.4f}")
