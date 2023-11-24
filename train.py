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
    size: str
    observe_camera: bool
    observe_raycast: bool
    observe_extra: bool
    n_parallel_envs: int
    eval_mode: bool
    wandb: bool
    from_checkpoint: Optional[Path]
    logdir: Optional[Path]
    dreamer_args: str
    debug: bool


def main():
    # CLI Configuration (aligns with Args dataclass)
    # TODO: Implement --episodes
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
        default=Path("auto"),
        help="Path to the AnimalAI executable.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "xlarge"],
        help="Size of the DreamerV3 agent.",
    )
    parser.add_argument(
        "--observe-raycast",
        default=False,
        action="store_true",
        help="Whether to let the agent observe the raycast rays.",
    )
    parser.add_argument(
        "--observe-extra",
        default=False,
        action="store_true",
        help="Whether to let the agent observe health, velocity, and global position.",
    )
    parser.add_argument(
        "--disable-camera",
        dest="observe_camera",
        default=True,
        action="store_false",
        help="Disable camera/pixel observations.",
    )
    parser.add_argument(
        "--n-parallel-envs",
        default=1,
        type=int,
        help="Number of parallel environments to run.",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode.",
    )

    # Parse and start
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))
    run(args)


def run(args: Args):
    # Validate CLI args
    if args.eval_mode:
        assert args.from_checkpoint, "Must provide a checkpoint to evaluate from."
    if args.from_checkpoint:
        ckpt_exists = args.from_checkpoint.exists()
        ckpt_is_file = args.from_checkpoint.is_file()
        assert ckpt_exists, f"Checkpoint not found: {args.from_checkpoint}."
        assert ckpt_is_file, f"Checkpoint must be a file but is not: {args.from_checkpoint}."  # fmt: skip
    if not args.observe_camera and not args.observe_raycast:
        raise ValueError("No observations: camera is disabled and raycasts are not enabled.")  # fmt: skip
    assert args.n_parallel_envs > 0, "Must have at least one environment."

    # Make logdir
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    if args.logdir is not None:
        logdir = args.logdir
    else:
        task_name = Path(args.task).stem
        runtype = "eval" if args.eval_mode else "training"
        runtype = "debug" if args.debug else runtype
        logdir = Path("./logdir/") / f"{runtype}-{date}-{task_name}"
    logdir.mkdir(parents=True)

    # Set up Python logging
    (logdir / "log.txt").touch()
    handler = logging.FileHandler(logdir / "log.txt")
    handler.setLevel(logging.INFO)
    format = "[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s"
    handler.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(handler)

    # Configure task
    if str(args.task) == "empty":
        task_path = None  # Will spawn an empty Arena
        logging.warning("Using an empty Arena.")
    else:
        assert args.task.exists(), f"Task file not found: {args.task}."
        task_path = args.task

    # Configure environment binary
    if str(args.env) == "auto":
        env_path = find_env_path(Path("./aai/"))
    else:
        assert args.env.exists(), f"AAI executable file not found: {args.env}."
        env_path = args.env
    logging.info(f"Using AAI executable file: {env_path}")

    # Log some specific stuff for reference
    logging.info(f"CLI Args: {args}")
    if task_path:
        shutil.copy(task_path, logdir / task_path.name)
    else:
        (logdir / "empty_task.yaml").touch()

    # Dreamer and AAI setup
    logging.info("Creating DreamerV3 and AAI Environment")
    agent_config = Glue.get_config(
        logdir,
        size=args.size,
        observe_camera=args.observe_camera,
        observe_raycast=args.observe_raycast,
        observe_extra=args.observe_extra,
        dreamer_args=args.dreamer_args,
        from_checkpoint=args.from_checkpoint,
        debug=args.debug,
    )
    agent_config.save(logdir / "dreamer_config.yaml")
    logger, step = Glue.get_loggers(logdir, agent_config, args.wandb)
    env = Glue.get_env(task_path, env_path, agent_config, args.n_parallel_envs)
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
        size: str,
        observe_camera: bool = True,
        observe_raycast: bool = False,
        observe_extra: bool = False,
        dreamer_args: str = "",
        from_checkpoint: Optional[Path] = None,
        debug: bool = False,
    ) -> embodied.Config:
        # See configs.yaml for all options.
        config = embodied.Config(dreamerv3.configs["defaults"])
        config = config.update(dreamerv3.configs[size])  # E.g. medium or xlarge
        config = config.update({
                "logdir": logdir,
                "run.train_ratio": 64,  # Same as dmlab
                "run.log_every": 60,  # seconds
                "batch_size": 16,
                "jax.prealloc": True,
        })  # fmt: skip

        # Decide which observations the MLP sees
        if not observe_raycast and not observe_extra:
            mlp_obs = "$^"
        else:
            keys = []
            keys += ["raycast"] if observe_raycast else []
            keys += ["extra"] if observe_extra else []
            mlp_obs = "(" + "|".join(keys) + ")"

        config = config.update({
                "encoder.mlp_keys": mlp_obs,
                "decoder.mlp_keys": mlp_obs,
                "encoder.cnn_keys": "image" if observe_camera else "$^",
                "decoder.cnn_keys": "image" if observe_camera else "$^",
        })  # fmt: skip

        config.update({"run.from_checkpoint": from_checkpoint or ""})
        config.update(dreamerv3.configs["debug"] if debug else {})
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
        task_path: Optional[Union[Path, str]],
        env_path: Union[Path, str],
        agent_config: embodied.Config,
        n_parallel_envs: int = 1,
    ):
        # Use a random port to avoid problems if a previous version exits slowly
        port = 5005 + random.randint(0, 1000)

        envs = []
        for idx in range(n_parallel_envs):
            idx_string = f" {str(idx + 1)}" if n_parallel_envs > 1 else ""
            logging.debug(f"Initializing AAI environment{idx_string}")
            aai_env = AnimalAIEnvironment(
                file_name=str(env_path),
                base_port=port,
                arenas_configurations=str(task_path) if task_path is not None else "",
                worker_id=port + idx,
                # Set pixels to 64x64 cause it has to be power of 2 for dreamerv3
                resolution=64,  # same size as Minecraft in DreamerV3
                useCamera=True,
                useRayCasts=True,
                no_graphics=False,  # Without graphics we get gray only observations.
            )
            logging.debug(f"Wrapping AAI environment{idx_string}")
            env = UnityToGymWrapper(
                aai_env,
                uint8_visual=True,
                allow_multiple_obs=True,  # Also provide health, velocity (x, y, z), and global position (x, y, z)
                flatten_branched=True,  # Necessary. Dreamerv3 doesn't support MultiDiscrete action space.
            )
            env = EnvCompatibility(env, render_mode="rgb_array")  # type: ignore
            env = AAItoDreamerObservationWrapper(env)
            env = from_gym.FromGym(env)
            if idx == 0:
                logging.info(f"Using observation space {env.obs_space}")
                logging.info(f"Using action space {env.act_space}")
            env = dreamerv3.wrap_env(env, agent_config)
            envs.append(env)

        env = embodied.BatchEnv(envs, parallel=False)
        return env


class AAItoDreamerObservationWrapper(gym.ObservationWrapper):  # type: ignore
    """
    Go from a tuple to dict observation space,
    and split the raycast and extra (health, velocity, position) observations.

    <https://www.gymlibrary.dev/api/wrappers/#observationwrapper>
    """

    def __init__(self, env: Env):
        super().__init__(env)
        tuple_obs_space: gym.spaces.Tuple = self.observation_space  # type: ignore

        # RGB image (dimensions might vary)
        image = tuple_obs_space[0]

        # Raycasts in a 1D array of 20 entries.
        mixed: gym.spaces.Box = tuple_obs_space[1]  # type: ignore # Raycast + extra together
        raycast_size = mixed.shape[0] - 7
        raycast = gym.spaces.Box(float("-inf"), float("+inf"), shape=(raycast_size,), dtype=float)  # fmt: skip

        # Health, velocity (x, y, z), and global position (x, y, z) in a 1D array of 7 entries.
        extra = gym.spaces.Box(float("-inf"), float("+inf"), shape=(7,), dtype=float)

        self.observation_space = gym.spaces.Dict(
            {"image": image, "raycast": raycast, "extra": extra}
        )

    def observation(self, observation):
        image, mix = observation
        extra = mix[-7:]
        raycast = mix[:-7]
        return {"image": image, "extra": extra, "raycast": raycast}


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


if __name__ == "__main__":
    main()
