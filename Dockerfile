# ROAD-SHIELD AI Engine
#
#   docker build -t road-shield .
#   docker run -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints" road-shield
#
# The checkpoints volume matters. Trained models and the SQLite ledger live
# there, and a container without it starts with no models and an empty ledger -
# which the /system page will tell you about rather than pretending otherwise.
#
# No CUDA, no PyTorch. The CNN runs on ONNX Runtime on the CPU, which is why
# this image is ~700 MB instead of ~6 GB and why it runs on any machine.

FROM python:3.11-slim

# OpenCV needs these; slim does not ship them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# BLAS thread caps. Without them scikit-learn's backend spawns a thread per
# core and, on a small container, runs out of memory during training.
ENV OPENBLAS_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Fails the healthcheck if the models did not load, not merely if the port is open.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,json,sys; \
d=json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=8).read()); \
sys.exit(0 if d.get('status')=='ONLINE' else 1)"

CMD ["python", "-m", "api.server"]
