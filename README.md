# DreamerV3 for the AnimalAI Environment

## Install

1. Install requirements with `pip install -r requirements.txt`.
   **Note:** Only Python 3.9 is supported. Make sure that is the installed version on your OS, Docker container or virtual environment.
2. Download the [AnimalAI environment](https://github.com/Kinds-of-Intelligence-CFI/animal-ai#quick-install-please-see-release-for-latest-version-of-aai-3). The scripts will look for the env at `./aai/env/{AnimalAI,AAI}.{x86_64,exe,app}`, and later at `./aai/env*/{AnimalAI,AAI}.{x86_64,exe,app}` where the lexicographically latest folder will be selected. There is also an `--env` flag to set the path manually.
3. Run tests to sanity check the installation.

   - `python tests/jaxtest.py`
     Most important thing to watch for is warnings about no GPUs being detected.
   - `CUDA_VISIBLE_DEVICES=0 python tests/dreamertest.py`
     You should see various logs being written to the terminal, and eventually messages about which scores the Agent gets for each episode. Abort manually if things look good.
   - `python aaitest.py`
     Spawns a random AnimalAI arena playable through the keyboard. Press C to switch camera. Abort manually if things look good.
   - `CUDA_VISIBLE_DEVICES=0 ./tests/integration.sh`
     Test integration of DreamerV3 and AnimalAI, by starting a small training run in debug mode. Abort manually if things look good.

## Usage

Example:

```shell
python train.py --task aai/configs/sanityGreenAndYellow.yml
```

Adapt the `train.py` (and everything else) to your liking.

Many things can be configured using flags and dreamer config options.
For the latter, see [configs.yml](https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat/blob/main/dreamerv3/configs.yaml), and specify them like this: `--dreamer-args "--run.steps 1e6"`.

### Running on headless servers

Use Xvfb, e.g. through `CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py --task aai/configs/sanityGreenAndYellow.yml --env aai/env3.1.3/AAI.x86_64`

Alternatively, if Xvfb and an X server are not installed, run with Docker (see below).

### Running in Docker (on a headless server)

We provide a Docker image that installs:

- NVIDIA GPU utilities (from the base image)
- Base utilities (e.g. ffmpeg for logging DreamerV3 gifs)
- an X server, Xvfb, and other graphical tools so AnimalaI's Unity can render to something.
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

which gives an interactive shell. Be sure to replace `$(pwd)` with something that works on your system (if not running Linux), or just use an absolute path. The `--gpu` flag is needed to pass through host GPUs to the container, but there are some requirements (documented [here](https://docs.docker.com/config/containers/resource_constraints/#gpu)). Test on CPU by adding the flag `--dreamer-args "--jax.platform cpu"` to `./tests/integration-docker.sh`.

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
