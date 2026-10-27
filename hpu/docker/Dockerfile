# syntax=docker/dockerfile:1
#
# Strata, Gaudi2 backend: Qwen3.8-Flash-Next on one Intel Gaudi2 (HL-225, 96 GB).
#
# Build (from the repository root; the engine and its TPC kernels are compiled here):
#   docker build -f hpu/docker/Dockerfile -t strata-gaudi2 .
#   docker build -f hpu/docker/Dockerfile -t strata-gaudi2 \
#     --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
#
# Run: hpu/docker/compose.yaml (or the docker run line in the README). The host needs the habanalabs driver
# 1.24.x; the container gets the card as /dev/accel/accel0 + /dev/accel/accel_controlD0, no special runtime.
#
# The SynapseAI release of the base image must match the host driver (1.24.x here). The TPC compiler
# (tpc-clang, package habanalabs-tools) is not in the base image, so it comes from Habana's apt repository.
ARG BASE=vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0:latest
FROM ${BASE}

ARG HABANA_TOOLS_VERSION=1.24.1-482
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public | gpg --dearmor -o /usr/share/keyrings/habana.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/habana.gpg] https://vault.habana.ai/artifactory/debian noble main" \
      > /etc/apt/sources.list.d/habana.list && \
    apt-get update && apt-get install -y --no-install-recommends habanalabs-tools=${HABANA_TOOLS_VERSION} && \
    rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple
WORKDIR /opt/strata
COPY requirements.txt /opt/strata/requirements.txt
RUN pip install --no-cache-dir --break-system-packages --index-url "${PIP_INDEX_URL}" -r requirements.txt

COPY . /opt/strata
# The engine finds its TPC kernel library at the absolute build path, so the build stays where it is compiled.
# CMake downloads llama.cpp 3cf03257 (only its ggml, to read GGUF); offline: --build-arg GGML_DIR=<checkout in context>.
ARG GGML_DIR=
RUN cmake -S hpu -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release ${GGML_DIR:+-DSTRATA_HPU_GGML_DIR=/opt/strata/${GGML_DIR}} && \
    cmake --build build-release -j"$(nproc)" --target strata-hpu strata_hpu_tpc && \
    rm -rf build-release/_deps/*-subbuild && \
    chmod +x hpu/docker/entrypoint.sh

ENV PYTHONPATH=/opt/strata/tools \
    STRATA_PORT=8080
EXPOSE 8080
ENTRYPOINT ["/opt/strata/hpu/docker/entrypoint.sh"]
