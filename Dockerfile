FROM tensorflow/serving:latest
 
COPY . /models
ENV MODEL_NAME=xss-detection