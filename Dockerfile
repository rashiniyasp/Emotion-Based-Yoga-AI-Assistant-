FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libegl1 \
    libegl-mesa0 \
    libgles2 \
    libgbm1 \
    libdrm2 \
    mesa-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Force software EGL rendering (no real GPU needed on HF Spaces free tier)
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV PYOPENGL_PLATFORM=osmesa

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Expose port for Hugging Face Spaces (Must be 7860)
EXPOSE 7860

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
