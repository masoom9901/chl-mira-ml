FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for OpenCV, Matplotlib, and other scientific python libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Add healthcheck to monitor streamlit state
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
