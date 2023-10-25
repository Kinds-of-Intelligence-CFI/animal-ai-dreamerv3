import os
# Make sure this is above the import of AnimalAIEnvironment
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # noqa

from typing import *
import random
import shutil
import shlex
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import logging
logging.basicConfig(  # noqa
    format='[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s',
    level=logging.INFO
)

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from gym.wrappers.compatibility import EnvCompatibility
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment

@dataclass
class Args:
    task: Path
    env: Path
    eval_mode: bool
    from_checkpoint: Optional[Path]
    logdir: Optional[Path]
    dreamer_args: str


def run(args: Args):
    assert args.from_checkpoint is not None if args.eval_mode else True, "Must provide a checkpoint to evaluate from."
    assert args.from_checkpoint.exists() if  args.from_checkpoint is not None else True, f"Checkpoint not found: {args.from_checkpoint}."
    assert args.from_checkpoint.is_file() if  args.from_checkpoint is not None else True, f"Checkpoint must be a file but is not: {args.from_checkpoint}."
    assert args.task.exists(), f"Task file not found: {args.task}."
    assert args.env.exists(), f"AAI executable file not found: {args.env}."

    task_path = args.task
    task_name = Path(args.task).stem

    # Configure logging
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    if args.logdir is not None:
        logdir = args.logdir
    else:
        logdir = Path("./logdir/") / f'{"eval" if args.eval_mode else "training"}-{date}-{task_name}'
    logdir.mkdir(parents=True)
    (logdir / 'log.txt').touch()
    handler = logging.FileHandler(logdir / 'log.txt')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s'))
    logging.getLogger().addHandler(handler)
    shutil.copy(task_path, logdir / task_path.name) # Copy task file to logdir for reference

    logging.info(f"Args: {args}")

    logging.info("Creating DreamerV3 config")
    dreamer_config, step, logger = get_dreamer_config(logdir, args.dreamer_args, args.from_checkpoint)
    dreamer_config = dreamer_config.update({
        'run.from_checkpoint': args.from_checkpoint or '',
    })
    dreamer_config.save(logdir / 'dreamer_config.yaml')

    logging.info(f"Creating AAI Dreamer Environment")
    env = get_aai_env(task_path, args.env, dreamer_config)

    logging.info("Creating DreamerV3 Agent")
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamer_config)

    replay = embodied.replay.Uniform(
        dreamer_config.batch_length, 
        dreamer_config.replay_size,
        logdir / 'replay')
    emb_config = embodied.Config(
        **dreamer_config.run, 
        logdir=dreamer_config.logdir,
        batch_steps=dreamer_config.batch_size * dreamer_config.batch_length)  # type: ignore
    if args.eval_mode:
        logging.info("Starting evaluation")
        embodied.run.eval_only(agent, env, logger, emb_config)
    else:
        logging.info("Starting training")
        embodied.run.train(agent, env, replay, logger, emb_config)


def get_dreamer_config(logdir: Path, dreamer_args: str = '', from_checkpoint: Optional[Path] = None):
    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['xlarge'])
    config = config.update({
        'logdir': logdir,
        'run.train_ratio': 64, # Same as dmlab
        'run.log_every': 60,  # Seconds
        'batch_size': 16,
        'jax.prealloc': True, # We have enough memory to allow focusing on speed.
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
          'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse(shlex.split(dreamer_args))

    step = embodied.Counter()

    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        embodied.logger.WandBOutput(wandb_init_kwargs={
          'project': 'dreamerv3-animalai',
          'name': logdir.name,
          'config': dict(config),
        }),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])

    return config, step, logger


def get_aai_env(task_path: Union[Path, str], env_path: Union[Path, str], dreamer_config):
    # Use a random port to avoid problems if a previous version exits slowly
    port = 5005 + random.randint(0, 1000)

    logging.info("Initializing AAI environment")
    aai_env = AnimalAIEnvironment(
        file_name=str(env_path),
        base_port=port,
        arenas_configurations=str(task_path),
        # Set pixels to 64x64 cause it has to be power of 2 for dreamerv3
        resolution=64, # same size as Minecraft in DreamerV3
    )
    logging.info("Wrapping AAI environment")
    env = UnityToGymWrapper(
        aai_env, uint8_visual=True,
        # allow_multiple_obs=True, # This crashes somewhere in one of the wrappers.
        flatten_branched=True)  # Necessary. Dreamerv3 doesn't support MultiDiscrete action space.
    env = EnvCompatibility(env, render_mode='rgb_array')  # type: ignore
    env = from_gym.FromGym(env, obs_key='image')
    logging.info(f"Using observation space {env.obs_space}")
    logging.info(f"Using action space {env.act_space}")
    env = dreamerv3.wrap_env(env, dreamer_config)
    # env = MonkeyPatchLen(env)  # Dreamerv3 expects env to have a __len__ method.
    env = embodied.BatchEnv([env], parallel=False)

    return env

class MonkeyPatchLen(embodied.core.base.Wrapper):
    def __len__(self):
        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=Path, required=True, help='Path to the task file.')
    parser.add_argument('--env', type=Path, required=True, help='Path to the AnimalAI executable.')
    parser.add_argument('--eval-mode', action='store_true', help='Run in evaluation mode. Make sure to also load a checkpoint.')
    parser.add_argument('--from-checkpoint', type=Path, help='Load a checkpoint to continue training or evaluate from.')
    parser.add_argument('--logdir', type=Path, help='Directory to save logs to.')
    parser.add_argument('--dreamer-args', type=str, default='', help='Extra args to pass to dreamerv3.')
    args_raw = parser.parse_args()

    args = Args(**vars(args_raw))

    run(args)