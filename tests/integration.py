import os
# Make sure this is above the import of AnimalAIEnvironment
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # noqa

from pathlib import Path
from typing import *
import dataclasses
import logging
from datetime import datetime
logging.basicConfig(  # noqa
    # format='[%(asctime)s] [%(levelname)-8s] [%(pathname)s] %(message)s',
    format='[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s',
    level=logging.INFO
)

import random

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from gym.wrappers.compatibility import EnvCompatibility

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment


def get_dreamer_config(run_logdir, args):
    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
        'logdir': run_logdir,
        'run.train_ratio': 64,
        'run.log_every': 30,  # Seconds
        'batch_size': 16,
        'jax.prealloc': False,
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })
    if args.cpu:
        config = config.update({
            'jax.platform': 'cpu',
        })
    config = embodied.Flags(config).parse()

    step = embodied.Counter()

    logdir = embodied.Path(config.logdir)
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(wandb_init_kwargs={
        #     'project': 'dreamerv3-animalai',
        #     'name': logdir.name,
        #     'config': dict(config),
        # }),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])

    return config, step, logger, logdir


def aai_env(task_path: Union[Path, str], env_path: Union[Path, str], dreamer_config, logdir):
    # use a random port to avoid problems if a previous version exits slowly
    port = 5005 + random.randint(0, 1000)

    logging.info("Initializing AAI environment")
    aai_env = AnimalAIEnvironment(
        file_name=str(env_path),
        base_port=port,
        arenas_configurations=str(task_path),
        # Set pixels to 64x64 cause it has to be power of 2 for dreamerv3
        resolution=64,
        # Don't enable when using visual observations, as they will be all gray. Maybe okay when raycasting.
        # If enable, also enable log_folder to prevent terminal spammed by "No graphics device" logs from Unity
        # no_graphics=True, 
        # log_folder=logdir,
    )
    logging.info("Applying UnityToGymWrapper")
    env = UnityToGymWrapper(
        aai_env, uint8_visual=True,
        # allow_multiple_obs=True, # This crashes somewhere in one of the wrappers.
        flatten_branched=True)  # Necessary. Dreamerv3 doesn't support MultiDiscrete action space.

    logging.info("Applying EnvCompatibility")
    env = EnvCompatibility(env, render_mode='rgb_array')  # type: ignore

    logging.info("Applying DreamerV3 FromGym")
    env = from_gym.FromGym(env, obs_key='image')
    logging.info(f"Using observation space {env.obs_space}")
    logging.info(f"Using action space {env.act_space}")

    logging.info("Wrapping DreamerV3 environment")
    env = dreamerv3.wrap_env(env, dreamer_config)
    logging.info("Creating BatchEnv")
    env = embodied.BatchEnv([env], parallel=False)

    return env


def main(task_config, env_path, args):

    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logdir = Path("./logdir/") / f'integration-test-{date}'

    logging.info("Creating DreamerV3 config")
    dreamer_config, step, logger, logdir = get_dreamer_config(logdir, args)
    logging.info(f"Creating AAI Dreamer Environment")
    env = aai_env(task_config, env_path, dreamer_config, logdir)

    logging.info("Creating DreamerV3 Agent")
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamer_config)
    replay = embodied.replay.Uniform(
        dreamer_config.batch_length, dreamer_config.replay_size, logdir / 'replay')
    args = embodied.Config(
        **dreamer_config.run, logdir=dreamer_config.logdir,
        batch_steps=dreamer_config.batch_size * dreamer_config.batch_length)  # type: ignore

    logging.info("Starting training")
    embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)

@dataclasses.dataclass
class Args:
    config: Optional[Path]
    env: Optional[Path]
    cpu: bool

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs='?', type=Path, default=None)
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
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

    main(config, env_path, args)
