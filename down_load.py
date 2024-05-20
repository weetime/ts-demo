import kagglehub

# Download latest version
path = kagglehub.model_download("google/mobilenet-v3/tfLite/large-075-224-classification")

print("Path to model files:", path)