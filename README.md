# DreamerV3 for the AnimalAI Environment

## Usage

1. Install (most) dependencies with `conda env create --prefix .venv -f environment.yml`.
2. Activate your conda env with `conda activate ./.venv`.
3. Install dreamerv3 separately (installing together makes conda download jax without GPU support)
   `pip install "git+https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat.git#egg=dreamerv3" numpy==1.21.2`
   Note: We explicitly ask for `numpy==1.21.1`, as pip overrides conda installed version (but others are not compatible with ml-agents).
4. Run `python tests/jaxtest.py` and `CUDA_VISIBLE_DEVICES=0 python tests/dreamertest.py` to sanity check the installation. We set only the 0'th (i.e. the first) GPU visible because DreamerV3 does not explicitly support multi GPU and we don't want to hog more than necessary. Not needed if there is only 1 GPU available.
5. Download the [AnimalAI environment](https://github.com/Kinds-of-Intelligence-CFI/animal-ai#quick-install-please-see-release-for-latest-version-of-aai-3). The current train script will look for the env at `./aai/env/AnimalAI.x86_64`, but there is an `--aai` flag available to look in different locations, or you can just modify the train script.
6. Test installation of AnimalAI by running `python aaitest.py`, which should spawn a random AnimalAI arena playable by you through the keyboard, and run the integration test `CUDA_VISIBLE_DEVICES=0 python tests/integration.py integration.py`. If things don't crash, they likely work. Also watch for the errors.
7. Adapt the `train.py` script to your training needs.

## References

- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project>
- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai>
- <https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat>
