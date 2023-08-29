# TODO: Currently not working
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /dreamerv3-animalai

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y git \
    && apt-get install -y software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xvfb
RUN apt-get update \
    && apt-get install -y xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash

# Install a python 3.9 environment
COPY environment.yml .
RUN conda env create --prefix .venv -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-p", ".venv", "/bin/bash", "-c"]

COPY requirements.txt .
RUN pip3 install --no-cache-dir --requirement requirements.txt

COPY aai/ aai/
COPY tests/ tests/
COPY train.py .

CMD [ "/bin/bash" ]