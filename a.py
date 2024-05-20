import tensorflow as tf
print("Is Metal available:", tf.config.experimental.list_physical_devices('GPU'))
