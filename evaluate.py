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
import fnmatch

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from gym.wrappers.compatibility import EnvCompatibility

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment

import from_gym_aai


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
    # if args.cpu:
    #     config = config.update({
    #         'jax.platform': 'cpu',
    #     })
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


def aai_env(task_path: Union[Path, str], env_path: Union[Path, str], step_csv_path, dreamer_config, logdir, multi_obs = False):
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
        aai_env, 
        uint8_visual=True,
        allow_multiple_obs=multi_obs, # This crashes somewhere in one of the wrappers.
        flatten_branched=True)  # Necessary. Dreamerv3 doesn't support MultiDiscrete action space.

    logging.info("Applying EnvCompatibility")
    env = EnvCompatibility(env, render_mode='rgb_array')  # type: ignore

    logging.info("Applying DreamerV3 FromGym")
    if multi_obs:
        env = from_gym_aai.FromGymAAI(env, obs_key='image', step_csv_path = step_csv_path, multi_obs = multi_obs)
    else:
        env = from_gym.FromGym(env, obs_key='image')
    logging.info(f"Using observation space {env.obs_space}")
    logging.info(f"Using action space {env.act_space}")

    logging.info("Wrapping DreamerV3 environment")
    env = dreamerv3.wrap_env(env, dreamer_config)
    logging.info("Creating BatchEnv")
    env = embodied.BatchEnv([env], parallel=False)

    return env

def find_yaml_files(directory):
    yaml_files = []
    task_names = []
    
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.yml') + fnmatch.filter(filenames, '*.yaml'):
            yaml_files.append(os.path.join(root, filename))
            task_names.append(filename)
    
    return yaml_files, task_names

def main(task_config, env_path, multi_obs, episode_name, args):

    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logdir = Path("./logdir/") / f'evaluation-{episode_name}-{date}'

    csv_path = logdir / f"{episode_name}.csv"

    logging.info("Creating DreamerV3 config")
    dreamer_config, step, logger, logdir = get_dreamer_config(logdir, args)
    dreamer_config = dreamer_config.update({
        'run.from_checkpoint': './logdir/integration-test-2023_11_04_11_37/checkpoint.ckpt',
    })
    logging.info(f"Creating AAI Dreamer Environment")
    env = aai_env(task_config, env_path, csv_path, dreamer_config, logdir, multi_obs=multi_obs)

    logging.info("Creating DreamerV3 Agent")
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamer_config)
    replay = embodied.replay.Uniform(
        dreamer_config.batch_length, dreamer_config.replay_size, logdir / 'replay')
    args = embodied.Config(
        **dreamer_config.run, logdir=dreamer_config.logdir,
        batch_steps=dreamer_config.batch_size * dreamer_config.batch_length)  # type: ignore

    logging.info(f"Starting evaluation on {episode_name}")
    #embodied.run.train(agent, env, replay, logger, args)
    embodied.run.eval_only(agent, env, logger, args)

@dataclasses.dataclass
class Args:
    configfolder: Optional[Path]
    env: Optional[Path]
    #ckptpath: Path

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configfolder", type=Path, default=None)
    parser.add_argument("--env", type=Path, default=None)
    #parser.add_argument("--ckptpath", type=Path, default=None)
    args_raw = parser.parse_args()

    eval_args = Args(**vars(args_raw))


    import glob
    if eval_args.configfolder is not None:
        config_file_paths, config_file_names = find_yaml_files(eval_args.configfolder)
    else:
        config_file_paths, config_file_names = find_yaml_files("./aai/configs") 

    if eval_args.env is None:
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
    else:
        env_path = eval_args.env

    # if eval_args.ckptpath is None:
    #     ckpt_path = "./logdir/integration-test-2023_11_04_11_37/checkpoint.ckpt"
    # else:
    #     ckpt_path = eval_args.ckptpath

    print(f"Using environment {env_path}.")
    print(f"Using configuration folder 'aai/configs/'.")
    print(f"Found {len(config_file_paths)}. Evaluating on each.")
    
    for file in range(len(config_file_paths)):
        config = config_file_paths[file]

        main(config, env_path, multi_obs = True, args = "--run.steps 1", episode_name=config_file_names[file])
