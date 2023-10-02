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

4. Adapt the `train.py` script to your training needs.

### Running on headless servers

Use Xvfb, e.g. through `CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py --env PATH`

## References

- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project>
- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai>
- <https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat>
