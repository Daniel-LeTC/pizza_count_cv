FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create Streamlit config directory
RUN mkdir -p .streamlit

# Create Streamlit config file
RUN echo '[server]' > .streamlit/config.toml && \
    echo 'maxUploadSize = 2048' >> .streamlit/config.toml && \
    echo 'maxMessageSize = 2048' >> .streamlit/config.toml && \
    echo 'enableXsrfProtection = false' >> .streamlit/config.toml

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY config/ ./config/
COPY core/ ./core/
COPY utils/ ./utils/

# Create directories
RUN mkdir -p models videos output

ENV PYTHONPATH=/app
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


