import os
# Make sure this is above the import of AnimalAIEnvironment
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # noqa

from pathlib import Path
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


def get_dreamer_config(run_logdir):
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
        #   'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse()

    step = embodied.Counter()

    logdir = embodied.Path(config.logdir)
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])

    return config, step, logger, logdir


def aai_env(task_path, dreamer_config, logdir):
    # use a random port to avoid problems if a previous version exits slowly
    port = 5005 + random.randint(0, 1000)

    # env_path = "./aai/env3.0.1/AAI_v3.0.1_build_linux_090422.x86_64"
    env_path = "./aai/env3.0.2/AAI3Linux.x86_64"

    logging.info("Initializing AAI environment")
    aai_env = AnimalAIEnvironment(
        file_name=env_path,
        base_port=port,
        arenas_configurations=task_path,
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


def main():
    task_config = "./aai/configs/synergysimple1.yml"

    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logdir = Path("./logdir/") / f'integration-test-{date}'

    logging.info("Creating DreamerV3 config")
    dreamer_config, step, logger, logdir = get_dreamer_config(logdir)
    logging.info(f"Creating AAI Dreamer Environment")
    env = aai_env(task_config, dreamer_config, logdir)

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


if __name__ == '__main__':
    main()
