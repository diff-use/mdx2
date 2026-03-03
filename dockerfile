FROM python:3.10-slim AS stage1

FROM mambaorg/micromamba:1.5.5 AS stage2

FROM debian:stable-slim AS final

RUN apt-get update && apt-get install -y ca-certificates git

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

# Install the local package and Prefect in editable mode within the environment
# NOTE: Prefect is not available in conda-forge, so we use pip.
# Pin to Prefect 2.20.25 to match the prefect-server image. Prefect 3.x uses a different API and is incompatible.
RUN /usr/local/bin/micromamba run -n mdx2-dev pip install -e . "prefect==2.20.25" && \
    /usr/local/bin/micromamba run -n mdx2-dev pip install git+https://github.com/FlexXBeamline/dials-extensions

EXPOSE 4200 8888

