FROM python:3.10-slim AS stage1

FROM mambaorg/micromamba:1.5.5 AS stage2

FROM debian:stable-slim AS final

RUN apt-get update && apt-get install -y ca-certificates

COPY --from=stage1 /usr/local /usr/local
COPY --from=stage2 /bin/micromamba /usr/local/bin/micromamba

ENV PATH="/usr/local/bin:/opt/conda/bin:$PATH"

# Ensure /opt/conda exists for micromamba to use
RUN mkdir -p /opt/conda

WORKDIR /home/dev

COPY env.yaml .

# Use micromamba to create the environment and install packages
RUN /usr/local/bin/micromamba create -f env.yaml -n mdx2-dev && \
    /usr/local/bin/micromamba install -y -n mdx2-dev nexpy jupyterlab jupyterlab-h5web dials xia2 wget tar -c conda-forge && \
    /usr/local/bin/micromamba clean --all --yes

COPY . .

# Install the local package in editable mode within the environment
RUN /usr/local/bin/micromamba run -n mdx2-dev pip install -e .

EXPOSE 8888

