# DreamerV3 for the AnimalAI Environment

## Usage

1. Install all dependencies with `conda env create --prefix .venv -f environment.yml`.
2. Activate your conda env with `conda activate ./.venv`.
3. Run all the tests in the `tests/` dir to sanity check the installation. I recommend starting with `python tests/jaxtest.py` and then moving on to `dreamertest.py`, followed by `aaitest.py` and lastly `integration.py`. If things don't crash, they work.
4. Adapt the `integration.py` script to your training needs.

## References

- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project>
- <https://github.com/Kinds-of-Intelligence-CFI/animal-ai>
- <https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat>
