import sys
import random
import os
import dataclasses
from pathlib import Path
from typing import *

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # noqa

from animalai.envs.environment import AnimalAIEnvironment

def load_config_and_play(configuration_file: str) -> None:
    """
    Loads a configuration file for a single arena and lets you play manually
    :param configuration_file: str path to the yaml configuration
    :return: None
    """
    # env_path = "./aai/env3.0.2/AAI3Linux.x86_64"
    env_path = "./aai/env3.1.2.exp/AAI.x86_64"
    env_path = "./aai/env3.1.2.exp.2.pre/AAI.x86_64"
    # env_path = "./aai/env3.0.1/AAI_v3.0.1_build_linux_090422.x86_64"
    # use a random port to avoid problems if a previous version exits slowly
    port = 5005 + random.randint(0, 1000)

    print("Initializing AAI environment")
    environment = AnimalAIEnvironment(
        file_name=env_path,
        base_port=port,
        # arenas_configurations=configuration_file,  # type: ignore
        # arenas_configurations="./aai/configs/paper/foragingTask/foragingTaskSpawnerTree.yml",
        arenas_configurations="./aai/configs/sanityGreenAndYellow.yml",
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

# If an argument is provided then assume it is path to a configuration and use that
# Otherwise load a random competition config.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs='?', type=Path, default=None)
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))

    if args.config is not None:
        config = args.config
    else:
        import glob
        config_files = list(glob.glob("./aai/configs/**/*.yml", recursive=True))
        config = Path(random.choice(config_files))
    
    print(F"Using configuration file {config}")
    load_config_and_play(configuration_file=str(config))

