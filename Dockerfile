# TODO: Currently not working
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /dreamerv3-animalai

# Add cleanup script for use after apt-get
RUN echo '#! /bin/sh\n\
env DEBIAN_FRONTEND=noninteractive apt-get autoremove --purge -y\n\
apt-get clean\n\
find /var/lib/apt/lists -type f -delete\n\
find /var/cache -type f -delete\n\
find /var/log -type f -delete\n\
exit 0\n\
' > /apt_cleanup && chmod +x /apt_cleanup

# Install base utilities
RUN apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        ca-certificates \
        wget \
        git && \
    /apt_cleanup

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
# ... activate conda env by default in interactive shell
RUN conda init bash && \
    echo "conda activate ./.venv" >> ~/.bashrc

# Install a python 3.9 environment
COPY environment.yml .
RUN conda env create --prefix .venv -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-p", ".venv", "/bin/bash", "-c"]

# Install python dependencies (in the conda environment)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install X and xvfb
RUN apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        xorg \
        xserver-xorg \
        xvfb && \
    /apt_cleanup

COPY aai/ aai/
COPY tests/ tests/
COPY train.py .

CMD [ "/bin/bash"]