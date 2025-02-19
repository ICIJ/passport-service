# syntax=docker/dockerfile:1
FROM python:3.10-slim-bullseye AS python-base

ENV HOME=/home/user
WORKDIR $HOME
RUN apt-get update && apt-get install -y curl

RUN curl -LsSf https://astral.sh/uv/0.5.6/install.sh | sh
ENV PATH="$HOME/.local/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

##### HTTP serivce
FROM python-base AS passport-service

ARG dbmate_arch
WORKDIR $HOME/src/app
RUN curl -fsSL -o /usr/local/bin/dbmate https://github.com/amacneil/dbmate/releases/download/v2.19.0/dbmate-linux-${dbmate_arch} \
    && chmod +x /usr/local/bin/dbmate
# Install deps first to optimize layer cache
RUN --mount=type=cache,target=~/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync -v --frozen --no-editable --no-install-project --extra http
# Then copy code
ADD uv.lock pyproject.toml README.md ./
ADD passport_service  ./passport_service/
# Then install service
RUN cd passport_service && uv sync -v --frozen --no-editable --extra http
RUN rm -rf ~/.cache/pip $(uv cache dir)

ENTRYPOINT ["uv", "run", "python", "-m", "passport_service"]

##### Worker base
FROM python-base AS worker-base
WORKDIR $HOME/src/app


##### Preprocessing worker
FROM worker-base AS preprocessing-worker
ARG n_workers
ENV N_PROCESSING_WORKERS $n_workers
# Install deps first to optimize layer cache
RUN --mount=type=cache,target=~/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync -v --frozen --no-editable --no-install-project --extra preprocessing
# Then copy code
ADD uv.lock pyproject.toml README.md ./
ADD passport_service  ./passport_service/
ADD scripts  ./scripts/
# Then install service
RUN cd passport_service && uv sync -v --frozen --no-editable --extra preprocessing
RUN rm -rf ~/.cache/pip $(uv cache dir)

ENTRYPOINT ["/home/user/src/app/scripts/preprocessing_entrypoint.sh"]


##### Inference worker
FROM worker-base AS inference-base
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 tesseract-ocr

FROM inference-base AS inference-worker
# TODO: fix that for ARM
RUN apt-get -y install wget software-properties-common
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && add-apt-repository contrib \
    && apt-get update
RUN apt-get -y install cuda-toolkit-12-3 cudnn9-cuda-12
# Install deps first to optimize layer cache
RUN --mount=type=cache,target=~/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync -v --frozen --no-editable --no-install-project --extra inference --extra gpu
# Then copy code
ADD uv.lock pyproject.toml README.md ./
ADD passport_service  ./passport_service/
# Then install service
RUN cd passport_service && uv sync -v --frozen --no-editable --extra inference --extra gpu
RUN rm -rf ~/.cache/pip $(uv cache dir)
ENTRYPOINT ["uv", "run", "python", "-m", "icij_worker", "workers", "start", "passport_service.app.app", "-g", "inference"]

FROM inference-base AS inference-worker-cpu
# Install deps first to optimize layer cache
RUN --mount=type=cache,target=~/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync -v --frozen --no-editable --no-install-project --extra inference --extra cpu
ADD uv.lock pyproject.toml README.md ./
ADD passport_service  ./passport_service/
RUN cd passport_service && uv sync -v --frozen --no-editable --extra inference --extra cpu
RUN rm -rf ~/.cache/pip $(uv cache dir)
ENTRYPOINT ["uv", "run", "python", "-m", "icij_worker", "workers", "start", "passport_service.app.app", "-g", "inference"]
