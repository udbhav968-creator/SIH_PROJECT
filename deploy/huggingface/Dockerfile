# ROAD-SHIELD live engine for a Hugging Face Space (Docker SDK, free CPU hardware).
#
# This Space holds only this file and README.md. At build time it clones the project from GitHub, so the
# models, code and bundled photographs are exactly what is on GitHub - nothing is uploaded by hand.
# To pick up new commits: Space -> Settings -> "Factory rebuild".
#
# It runs in public-demo mode (ROAD_SHIELD_PUBLIC=1): analysis endpoints are open and store nothing, write
# endpoints are locked, requests cannot name server files outside datasets/, and each visitor gets
# 20 model requests per minute.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Spaces as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user

ARG REPO=https://github.com/udbhav968-creator/SIH_PROJECT.git
ARG BRANCH=master
RUN git clone --depth 1 --branch ${BRANCH} ${REPO} app && rm -rf app/.git
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r requirements.txt

ENV ROAD_SHIELD_PUBLIC=1 \
    ROAD_SHIELD_RATE_LIMIT=20 \
    ROAD_SHIELD_MAX_BODY_MB=12 \
    ROAD_SHIELD_WRITABLE_DIR=/tmp/road_shield \
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["python", "-m", "api.server", "7860"]
