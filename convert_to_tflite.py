import tensorflow as tf
import tensorflow_hub as hub

class Model(tf.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
    def __call__(self, x):
        return self.hub_layer(x)

# 创建模型实例
model = Model()

# 将模型转换为 TensorFlow Lite 格式
converter = tf.lite.TFLiteConverter.from_concrete_functions([model.__call__.get_concrete_function()])
tflite_model = converter.convert()

# 保存转换后的模型
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TensorFlow Lite 模型已保存为 model.tflite")
