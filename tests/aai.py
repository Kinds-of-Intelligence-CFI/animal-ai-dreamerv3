import sys
import random
import os

from animalai.envs.environment import AnimalAIEnvironment

def load_config_and_play(configuration_file: str) -> None:
    """
    Loads a configuration file for a single arena and lets you play manually
    :param configuration_file: str path to the yaml configuration
    :return: None
    """
    env_path = "./aai/env/AAI3Linux.x86_64" # TODO
    port = 5005 + random.randint(
        0, 1000
    )  # use a random port to avoid problems if a previous version exits slowly

    print("Initializaing AAI environment")
    environment = AnimalAIEnvironment(
        file_name=env_path,
        base_port=port,
        arenas_configurations=configuration_file, # type: ignore
        play=True,
    )

    # Run the environment until signal to it is lost
    try:
        while environment._process: # type: ignore
            continue
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()


# If an argument is provided then assume it is path to a configuration and use that
# Otherwise load a random competition config.
if __name__ == "__main__":
    if len(sys.argv) > 1:
        config = sys.argv[1]
    else:
        config_folder = "./aai/configs/"
        config_files = os.listdir(config_folder)
        rand_idx = random.randint(0, len(config_files) - 1)
        config = config_folder + config_files[rand_idx]
        print(F"Using configuration file {config}")
    
    load_config_and_play(configuration_file=config)