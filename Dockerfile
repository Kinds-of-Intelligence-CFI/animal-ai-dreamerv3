FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /dreamerv3-animalai

# Add cleanup script for use after apt-get (and run cleanup to fix e.g. list issues)
RUN echo '#! /bin/sh\n\
env DEBIAN_FRONTEND=noninteractive apt-get autoremove --purge -y\n\
apt-get clean\n\
find /var/lib/apt/lists -type f -delete\n\
find /var/cache -type f -delete\n\
find /var/log -type f -delete\n\
exit 0\n\
' > /apt_cleanup && chmod +x /apt_cleanup && /apt_cleanup

# TODO: Remove. Temporary issues with ubuntu mirrors, using (local) italian mirror for now.
RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/it.archive.ubuntu.com/g' /etc/apt/sources.list

# Install base utilities
# ffmpeg is needed for logging DreamerV3 videos/gifs.
# git is needed for pip installs of git repos.
# The rest could maybe be removed.
RUN apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        ca-certificates \
        ffmpeg \
        wget \
        git && \
    /apt_cleanup

# Install X and xvfb
RUN apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        xorg \
        xserver-xorg \
        xvfb && \
    /apt_cleanup

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
# ... and activate conda env by default in interactive shell
RUN conda init bash && \
    echo "conda activate /dreamerv3-animalai/.venv" >> ~/.bashrc

# Install a python 3.9 environment
COPY environment.yml .
RUN conda env create --prefix .venv -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-p", "/dreamerv3-animalai/.venv", "/bin/bash", "-c"]

# Install python dependencies (in the conda environment)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# The ENTRYPOINT specified by NVIDIA's docker image is a bash wrapper 
# that prints some NVIDIA stuff, but also whether GPU drivers are detected (which is usefull).
# We don't override it.