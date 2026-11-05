# syntax=docker/dockerfile:1.6
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:kimi-k31@sha256:88175ddf4f59944ccbb24c6b60c1bc535a160be3c4f09a3b08702c5e301a599e
ARG SIDECAR_IMAGE=ghcr.io/tinfoilsh/inference-sidecar@sha256:65ce23d6560c46a1e8614ede187fcbf9798b267aa33878905b4872404787f47d
FROM ${SIDECAR_IMAGE} AS sidecar
FROM ${VLLM_BASE_IMAGE}

ENV VLLM_DISABLE_UVA=1 \
    VLLM_KIMI_K3_LATENT_MOE_TAIL_FUSION=0

COPY patches/ /tmp/k31-patches/
RUN python3 /tmp/k31-patches/apply-patches.py && \
    python3 -c "from importlib.metadata import version; assert version('vllm') == '0.1.dev24216+g266e88c05'" && \
    rm -rf /tmp/k31-patches

COPY validation/ /tmp/k31-validation/
RUN python3 -m venv --system-site-packages /tmp/k31-validation-venv && \
    /tmp/k31-validation-venv/bin/python -m unittest discover -s /tmp/k31-validation -p 'test_*.py' -v && \
    rm -rf /tmp/k31-validation /tmp/k31-validation-venv

COPY --from=sidecar /inference-sidecar /opt/tinfoil/inference-sidecar
ENTRYPOINT ["/opt/tinfoil/inference-sidecar", "vllm", "serve"]
