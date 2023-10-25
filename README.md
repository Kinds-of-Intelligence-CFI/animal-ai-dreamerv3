# DreamerV3 for the AnimalAI Environment

## Usage

1. Install requirements with `pip install --requirement requirements.txt`.
   **Note:** Only Python 3.9 is supported. Make sure that is the installed version on your OS, Docker container or virtual environment.
2. Download the [AnimalAI environment](https://github.com/Kinds-of-Intelligence-CFI/animal-ai#quick-install-please-see-release-for-latest-version-of-aai-3). The test scripts will look for the env at `./aai/env*/{AnimalAI,AAI}.{x86_64,exe,app}`, where the lexicographically latest folder will be selected, but there is an `--env` flag to set the path manually. For the train script, the `--env` flag is required. Feel free to just modify the scripts.
3. Run tests to sanity check the installation.

   - `python tests/jaxtest.py`
     Most important thing to watch for is warnings about no GPUs being detected.
   - `CUDA_VISIBLE_DEVICES=0 python tests/dreamertest.py`
     You should see various logs being written to the terminal, and eventually messages about which scores the Agent gets for each episode.
   - `python aaitest.py`
     Spawns a random AnimalAI arena playable through the keyboard. Press C to switch camera.
   - `CUDA_VISIBLE_DEVICES=0 python tests/integration.py`
     Test integration of DreamerV3 and AnimalAI. You should see the same things pass as with `dreamertest.py`

     We set only the 0'th (i.e. the first) GPU visible because DreamerV3 does not explicitly support multi GPU and we don't want to hog more resources than necessary. This is of course not needed if there is only 1 GPU available.

4. Adapt the `train.py` script to your training needs, e.g. disabling wandb logging.

### Running on headless servers

Use Xvfb, e.g. through `CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py --task aai/configs/sanityGreenAndYellow.yml --env aai/env3.1.3/AAI.x86_64`

Alternatively, if Xvfb and an X server are not installed, run with Docker (see below).

### Running in Docker (on a headless server)

We provide a Docker image that installs:

- NVIDIA GPU utilities (from the base image)
- Base utilities (e.g. ffmpeg for logging DreamerV3 gifs)
- an X server, Xvfb, and other graphical tools so AnimalaI's Unity can render to something
- Miniconda, and the conda environment specified in this repository.
- All python dependencies from requirements.txt.

The image is published at [woutschellaert/dreamerv3-animalai
](https://hub.docker.com/r/woutschellaert/dreamerv3-animalai).

**Note: The Docker image does not contain the AnimalAI Unity environment. Download the Linux build (as in step 2 above) and mount it to the container when executing `docker run`.**

If using wandb logging, be sure to add your API key to `.env`.

You can run the container like this:

```shell
$ docker run -it --rm \
  --env-file .env \
  --gpus '"device=0"' \
  -v $(pwd):/dreamerv3-animalai/mnt/ \
  --workdir /dreamerv3-animalai/mnt/ \
  dreamerv3-animalai
  woutschellaert/dreamerv3-animalai
```

which gives an interactive shell. Be sure to replace `$(pwd)` with something that works on your system, or just use an absolute path. The `--gpu` flag is needed to pass through host GPUs to the container, but there are some requirements (documented [here](https://docs.docker.com/config/containers/resource_constraints/#gpu)). We recommend testing with Dreamer in CPU mode first with the integration test `--cpu` flag.

You can also execute commands directly:

```shell
$ docker run --rm -it \
  --env-file .env \
  --gpus '"device=0"' \
  -v $(pwd):/dreamerv3-animalai/mnt/ \
  --workdir /dreamerv3-animalai/mnt/ \
  dreamerv3-animalai \
  ./tests/integration-docker.sh
```

**Note: There are some Docker bugs that prevent executing commands directly (i.e. they hang), but an easy workaround is to wrap the command in a script as demonstrated above. Other workarounds are described [here](https://stackoverflow.com/questions/41130240/docker-command-wont-work-unless-i-open-an-interactive-bash-terminal.)**

## References

- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project>
- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai>
- <https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat>
