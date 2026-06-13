FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libgles2-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Expose port for Hugging Face Spaces (Must be 7860)
EXPOSE 7860

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
