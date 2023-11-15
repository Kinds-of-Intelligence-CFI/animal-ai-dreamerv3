import random
import dataclasses
from pathlib import Path
from typing import Optional

# Make sure this is above the import of AnimalAIEnvironment
from animalai.envs.environment import AnimalAIEnvironment  # noqa: E402


def load_config_and_play(configuration_file: str, env_path: str) -> None:
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


def find_env_path(base: Path) -> Path:
    """
    Look for the latest version of the AAI.
    """
    error_msg = (
        "Could not automatically find any AAI environment binaries.\n\n"
        "We look in $BASE$/env/ for files matching {AAI,AnimalAI}.{x86_64,exe,app}, e.g. AAI.x86_64. "
        "Afterward, we look in the folders matching '$BASE$/env*/', "
        " taking the one that is lexicographically last, e.g. '$BASE$/env3.1.3/'.\n\n"
        "You can also specify the path exactly with the --env argument."
    ).replace("$BASE$", str(base))

    # Select folder
    env_folders = sorted(base.glob("env*"))
    if (base / "env/").exists():
        env_folders.append(base / "env/")
    if len(env_folders) == 0:
        reason = f"Could not find any folders matching {str(base)}/env*/"
        raise FileNotFoundError(f"{error_msg}\n\nReason: {reason}")
    env_folder = env_folders[-1]

    # Look for binary in selected folder
    # We brace expand manually because glob does not support it.
    binaries = [
        bin
        for bin_name in ["AAI", "AnimalAI"]
        for ext in ["x86_64", "exe", "app"]
        for bin in env_folder.glob(f"{bin_name}.{ext}")
    ]
    if len(binaries) == 0:
        reason = f"Could not find any AAI binaries in {env_folder}."
        raise FileNotFoundError(f"{error_msg}\n\nReason: {reason}")

    return binaries[0]


@dataclasses.dataclass
class Args:
    config: Optional[Path]
    env: Optional[Path]


# If an argument is provided then assume it is path to a configuration and use that
# Otherwise load a random competition config.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", type=Path, default=None)
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
        env_path = find_env_path(Path("./aai/"))

    print(f"Using environment {env_path}.")
    print(f"Using configuration file {config}.")
    load_config_and_play(
        configuration_file=str(config),
        env_path=str(env_path),
    )
