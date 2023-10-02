import sys
import random
import os
import dataclasses
from pathlib import Path
from typing import *

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # noqa

from animalai.envs.environment import AnimalAIEnvironment

def load_config_and_play(
        configuration_file: str,
        env_path: str
) -> None:
    """
    Loads a configuration file for a single arena and lets you play manually
    :param configuration_file: str path to the yaml configuration
    :return: None
    """
    # use a random port to avoid problems if a previous version exits slowly
    port = 5005 + random.randint(0, 1000)

    print("Initializing AAI environment")
    environment = AnimalAIEnvironment(
        file_name=env_path,
        base_port=port,
        arenas_configurations=configuration_file,
        # no_graphics=True,
        play=True,
    )

    # Run the environment until signal to it is lost
    try:
        while environment._process:  # type: ignore
            continue
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()

@dataclasses.dataclass
class Args:
    config: Optional[Path]
    env: Optional[Path]

# If an argument is provided then assume it is path to a configuration and use that
# Otherwise load a random competition config.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs='?', type=Path, default=None)
    parser.add_argument("--env", type=Path, default=None)
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))

    if args.config is not None:
        config = args.config
    else:
        import glob
        config_files = list(glob.glob("./aai/configs/**/*.yml", recursive=True))
        config = Path(random.choice(config_files))
    
    if args.env is not None:
        env_path = args.env
    else:
        # Look for latest version of AAI
        error_msg = "This script will look for the latest version of AAI in the ./aai folder " \
                    "by matching ./aai/env*/ where the folder with the " \
                    "lexically last value for * is used." \
                    "In that folder it will look for {AAI,AnimalAI}.{x86_64,exe,app}." \
                    "You also specify the path with the --env argument."
        env_folders = sorted(Path("./aai/").glob("env*"))
        assert len(env_folders) > 0, f"Could not find any AAI environments matching ./aai/env*/. \n{error_msg}"

        # We brace expand manually because glob does not support it.
        env_bins = [
            bin
            for bin_name in ["AAI", "AnimalAI"]
            for ext in ["x86_64", "exe", "app"]
            for bin in env_folders[-1].glob(f"{bin_name}.{ext}")
        ]
        assert len(env_bins) > 0, f"Could not find any AAI binaries in {env_folders[-1]}. \n{error_msg}"
        env_path = env_bins[0]

    print(f"Using environment {env_path}.")
    print(f"Using configuration file {config}.")
    load_config_and_play(
        configuration_file=str(config),
        env_path=str(env_path),
    )

