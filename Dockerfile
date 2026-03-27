FROM python:3.9-slim

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric (CPU)
RUN pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# Project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

# Generate sample data on build
RUN python scripts/generate_sample_data.py

CMD ["bash"]
