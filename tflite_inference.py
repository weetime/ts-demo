import tensorflow as tf
from PIL import Image
import numpy as np
import json

def load_model(model_path):
    # 加载 TensorFlow Lite 模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, input_size):
    # 预处理输入图像
    image = Image.open(image_path).resize(input_size).convert('RGB')
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    return input_data

def predict(interpreter, input_data):
    # 获取输入和输出张量
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 将数据提供给模型
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 运行推理
    interpreter.invoke()

    # 获取模型输出
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    return predicted_class

def load_labels(label_path):
    # 加载标签文件
    with open(label_path, "r") as f:
        labels = json.load(f)
    return [labels[str(i)][1] for i in range(len(labels))]

if __name__ == "__main__":
    model_path = "mobilenet-v3.tflite"
    image_path = "fengzheng.jpeg"
    label_path = "imagenet_class_index.json"
    input_size = (224, 224)  # 根据你的模型输入大小进行调整

    # 加载模型
    interpreter = load_model(model_path)

    # 预处理图像
    input_data = preprocess_image(image_path, input_size)

    # 进行预测
    predicted_class = predict(interpreter, input_data)

    # 加载标签
    labels = load_labels(label_path)

    # 输出预测结果
    print(f"Predicted class: {predicted_class}, Label: {labels[predicted_class]}")
