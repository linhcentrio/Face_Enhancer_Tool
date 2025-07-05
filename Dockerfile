# Base image with CUDA support
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev build-essential \
    libgl1-mesa-glx libglib2.0-0 ffmpeg wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    onnxruntime-gpu==1.17.1 \
    opencv-python==4.8.0.76 \
    numpy==1.24.4 \
    tqdm==4.67.1 \
    requests==2.28.1 \
    scikit-image==0.22.0 \
    Pillow==10.0.0 \
    scipy==1.11.4 \
    imutils==0.5.4 \
    librosa==0.10.1 \
    numba==0.58.1 \
    runpod

# Create required directories
RUN mkdir -p /app/enhancers/GFPGAN /app/faceID /app/outputs /app/temp

# Download necessary ONNX model files
RUN wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx" \
    -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx && \
    wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx" \
    -O /app/faceID/recognition.onnx && \
    wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx" \
    -O /app/faceID/arcface_w600k_r50.onnx

# Copy all application files
COPY . /app/

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose port for RunPod
EXPOSE 8000

# Command to run the handler
CMD ["python3", "rp_handler.py"]
