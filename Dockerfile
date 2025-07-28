# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system packages for OpenCV and other dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all local files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit \
    streamlit-lottie \
    opencv-python \
    torch \
    torchvision \
    torchaudio \
    numpy \
    matplotlib \
    pandas \
    Pillow \
    ultralytics \
    seaborn \
    scikit-learn \
    tqdm

# Run your Streamlit app
CMD ["streamlit", "run", "app.py"]
