# DreamerV3 for the AnimalAI Environment

## Usage

1. Install (most) dependencies with `conda env create --prefix .venv -f environment.yml`.
2. Activate your conda env with `conda activate ./.venv`.
3. Install dreamerv3 separately (installing together makes conda download jax without GPU support)
   `pip install "git+https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat.git#egg=dreamerv3" numpy==1.21.2`
   Note: We explicitly ask for `numpy==1.21.1`, as pip overrides conda installed version (but others are not compatible with ml-agents).
4. Run all the tests in the `tests/` dir to sanity check the installation. It is recommended starting with `python tests/jaxtest.py` and then moving on to `dreamertest.py`, followed by `aaitest.py` and lastly `integration.py`. If things don't crash, they work. Also watch for the errors. You can set `no_graphics=True` if working in a headless environment (but expect many warning messages from Unity).
5. Adapt the `integration.py` script to your training needs.

## References

- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project>
- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai>
- <https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat>
