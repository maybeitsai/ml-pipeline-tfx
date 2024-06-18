# Use the official TensorFlow image as a base image
FROM tensorflow/tfx:latest

# Set the working directory
WORKDIR /pipeline

# Copy the pipeline and model files to the container
COPY . /pipeline

# Install additional dependencies
RUN pip install --upgrade pip

# Set environment variables
ENV PIPELINE_NAME=xss-pipeline
ENV DATA_ROOT=/pipeline/data
ENV METADATA_PATH=/pipeline/metadata/${PIPELINE_NAME}/metadata.db
ENV PIPELINE_ROOT=/pipeline/pipelines/${PIPELINE_NAME}
ENV SERVING_MODEL_DIR=/pipeline/serving_model_dir/xss-detection-model