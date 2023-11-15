import logging
import random
import shutil
import shlex
import argparse
from typing import Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import gym
import gym.spaces
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from gym.wrappers.compatibility import EnvCompatibility
from gym import Env

# Make sure this is above the import of AnimalAIEnvironment
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s",
    level=logging.INFO,
)
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper  # noqa: E402
from animalai.envs.environment import AnimalAIEnvironment  # noqa: E402


@dataclass
class Args:
    task: Path
    env: Path
    eval_mode: bool
    wandb: bool
    from_checkpoint: Optional[Path]
    logdir: Optional[Path]
    dreamer_args: str


def main():
    # CLI Configuration (aligns with Args dataclass)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to the task file.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        required=True,
        help="Path to the AnimalAI executable.",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Run in evaluation mode. Make sure to also load a checkpoint.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log to Weights & Biases.",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=Path,
        help="Load a checkpoint to continue training or evaluate from.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        help="Directory to save logs to.",
    )
    parser.add_argument(
        "--dreamer-args",
        type=str,
        default="",
        help="Extra args to pass to dreamerv3.",
    )

    # Parse and start
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))
    run(args)


def run(args: Args):
    # Validate CLI args
    assert args.task.exists(), f"Task file not found: {args.task}."
    assert args.env.exists(), f"AAI executable file not found: {args.env}."
    if args.eval_mode:
        assert args.from_checkpoint, "Must provide a checkpoint to evaluate from."
    if args.from_checkpoint:
        ckpt_exists = args.from_checkpoint.exists()
        ckpt_is_file = args.from_checkpoint.is_file()
        assert ckpt_exists, f"Checkpoint not found: {args.from_checkpoint}."
        assert ckpt_is_file, f"Checkpoint must be a file but is not: {args.from_checkpoint}."  # fmt: skip

    task_path = args.task
    task_name = Path(args.task).stem

    # Configure logging
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    if args.logdir is not None:
        logdir = args.logdir
    else:
        runtype = "eval" if args.eval_mode else "training"
        logdir = Path("./logdir/") / f"{runtype}-{date}-{task_name}"
    logdir.mkdir(parents=True)
    (logdir / "log.txt").touch()
    handler = logging.FileHandler(logdir / "log.txt")
    handler.setLevel(logging.INFO)
    format = "[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s"
    handler.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(handler)
    shutil.copy(task_path, logdir / task_path.name)  # For reference

    logging.info(f"CLI Args: {args}")

    # Dreamer and AAI setup
    logging.info("Creating DreamerV3 and AAI Environment")
    agent_config = Glue.get_config(logdir, args.dreamer_args, args.from_checkpoint)  # fmt: skip
    agent_config.save(logdir / "dreamer_config.yaml")
    logger, step = Glue.get_loggers(logdir, agent_config, args.wandb)
    env = Glue.get_env(task_path, args.env, agent_config)
    agent, replay, run_args = Glue.get_agent(env, step, agent_config, logdir)

    # Run the agent
    if args.eval_mode:
        logging.info("Starting evaluation")
        embodied.run.eval_only(agent, env, logger, run_args)
    else:
        logging.info("Starting training")
        embodied.run.train(agent, env, replay, logger, run_args)

    # Close the environment
    logging.info("Closing environment")
    env.close()
    logging.info("Environment closed.")


class Glue:
    @staticmethod
    def get_config(
        logdir: Path,
        dreamer_args: str = "",
        from_checkpoint: Optional[Path] = None,
    ) -> embodied.Config:
        # See configs.yaml for all options.
        config = embodied.Config(dreamerv3.configs["defaults"])
        config = config.update(dreamerv3.configs["xlarge"])
        config = config.update(
            {
                "logdir": logdir,
                "run.train_ratio": 64,  # Same as dmlab
                "run.log_every": 60,  # Seconds
                "batch_size": 16,
                "jax.prealloc": True,  # We have enough memory to allow focusing on speed.
                "encoder.mlp_keys": "$^",
                "decoder.mlp_keys": "$^",
                "encoder.cnn_keys": "image",
                "decoder.cnn_keys": "image",
                # 'jax.platform': 'cpu',
            }
        )
        config.update(
            {
                "run.from_checkpoint": from_checkpoint or "",
            }
        )
        config = embodied.Flags(config).parse(shlex.split(dreamer_args))
        return config

    @staticmethod
    def get_loggers(
        logdir: Path,
        agent_config: embodied.Config,
        wandb: bool = True,
    ):
        step = embodied.Counter()
        loggers = [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            # embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.MLFlowOutput(logdir.name),
        ]
        if wandb:
            wandblogger = embodied.logger.WandBOutput(
                wandb_init_kwargs={
                    "project": "dreamerv3-animalai",
                    "name": logdir.name,
                    "config": dict(agent_config),
                }
            )
            loggers.append(wandblogger)
        return embodied.logger.Logger(step, loggers), step

    @staticmethod
    def get_agent(
        env: embodied.Env,
        step: embodied.Counter,
        agent_config: embodied.Config,
        logdir: Path,
    ):
        agent = dreamerv3.Agent(env.obs_space, env.act_space, step, agent_config)
        replay = embodied.replay.Uniform(
            agent_config.batch_length, agent_config.replay_size, logdir / "replay"
        )
        run_args = embodied.Config(
            **agent_config.run,
            logdir=agent_config.logdir,
            batch_steps=agent_config.batch_size * agent_config.batch_length,  # type: ignore
        )
        return agent, replay, run_args

    @staticmethod
    def get_env(
        task_path: Union[Path, str],
        env_path: Union[Path, str],
        agent_config: embodied.Config,
    ):
        # Use a random port to avoid problems if a previous version exits slowly
        port = 5005 + random.randint(0, 1000)

        logging.info("Initializing AAI environment")
        aai_env = AnimalAIEnvironment(
            file_name=str(env_path),
            base_port=port,
            arenas_configurations=str(task_path),
            # Set pixels to 64x64 cause it has to be power of 2 for dreamerv3
            resolution=64,  # same size as Minecraft in DreamerV3
        )
        logging.info("Wrapping AAI environment")
        env = UnityToGymWrapper(
            aai_env,
            uint8_visual=True,
            allow_multiple_obs=True,  # Also provide health, velocity (x, y, z), and global position (x, y, z)
            flatten_branched=True,  # Necessary. Dreamerv3 doesn't support MultiDiscrete action space.
        )
        env = EnvCompatibility(env, render_mode="rgb_array")  # type: ignore
        env = MultiObsWrapper(env)
        env = from_gym.FromGym(env)
        logging.info(f"Using observation space {env.obs_space}")
        logging.info(f"Using action space {env.act_space}")
        env = dreamerv3.wrap_env(env, agent_config)
        env = embodied.BatchEnv([env], parallel=False)

        return env


class MultiObsWrapper(gym.ObservationWrapper):  # type: ignore
    """
    Go from tuple to dict observation space.

    <https://www.gymlibrary.dev/api/wrappers/#observationwrapper>
    """

    def __init__(self, env: Env):
        super().__init__(env)
        tuple_obs_space: gym.spaces.Tuple = self.observation_space  # type: ignore
        self.observation_space = gym.spaces.Dict(
            {
                # RGB image
                "image": tuple_obs_space[0],
                # Health, velocity (x, y, z), and global position (x, y, z)
                # in a 1D array of 7 entries.
                "extra": tuple_obs_space[1],
            }
        )

    def observation(self, observation):
        image, extra = observation
        return {"image": image, "extra": extra}


if __name__ == "__main__":
    main()
