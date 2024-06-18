# Use the official TensorFlow Serving image as a base image
FROM tensorflow/serving:latest

# Copy the model files to the correct directory structure within the container
COPY serving_model_dir/xss-detection-model /models/xss-detection-model

# Set environment variables
ENV MODEL_NAME=xss-detection-model
ENV MODEL_BASE_PATH=/models