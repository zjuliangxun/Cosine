import monkey_patch
import torch
import numpy as np
import hydra
import pyrootutils
from omegaconf import DictConfig
from setproctitle import setproctitle
import utils.config
from omegaconf import OmegaConf

import utils.logger
from utils.config import set_np_formatting, set_seed, parse_sim_params, parse_phys_eigen

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from agent import amp_agent, parkour_agent
from agent import amp_players
from model import amp_models
from model import amp_network_builder, parkour_network_builder

from sim.vec_task_wrappers import VecTaskPythonWrapper
from sim.parkour_single import ParkourSingle
from sim.humanoid_amp import HumanoidAMPTask

import datetime

try:
    import wandb
except:
    wandb = None
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
torch.set_float32_matmul_precision("medium")


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and "consecutive_successes" in infos:
                cons_successes = infos["consecutive_successes"].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and "successes" in infos:
                successes = infos["successes"].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar("successes/consecutive_successes/mean", mean_con_successes, frame)
            self.writer.add_scalar("successes/consecutive_successes/iter", mean_con_successes, epoch_num)
            self.writer.add_scalar("successes/consecutive_successes/time", mean_con_successes, total_time)
        return


cfg_raw = None


def create_rlgpu_env(**kwargs):

    if cfg_complete["config"]["use_gpu"]:
        device_type = "cuda"
    else:
        device_type = "cpu"
    device_id = cfg_complete["rank"]
    sim_params = parse_sim_params(cfg_complete)

    cfg_complete["environment"]["env"]["seed"] = cfg_complete["seed"]
    try:
        task = eval(cfg_complete["environment"]["task"])(
            cfg=cfg_complete["environment"],
            sim_params=sim_params,
            physics_engine=parse_phys_eigen(cfg_complete["config"]["physics_engine"]),
            device_type=device_type,
            device_id=device_id,
            headless=cfg_complete["config"]["headless"],
        )
    except NameError as e:
        print(e)

    rl_device = "cuda:" + str(device_id) if cfg_complete["config"]["use_gpu"] else "cpu"
    env = VecTaskPythonWrapper(
        task,
        rl_device,
        cfg_complete["config"].get("clip_observations", np.inf),
        cfg_complete["config"].get("clip_actions", 1.0),
    )

    print("num_envs: {:d}".format(env.num_envs))
    print("num_actions: {:d}".format(env.num_actions))
    print("num_obs: {:d}".format(env.num_obs))
    print("num_states: {:d}".format(env.num_states))

    frames = kwargs.pop("frames", 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


env_configurations.register(
    "rlgpu", {"env_creator": lambda **kwargs: create_rlgpu_env(**kwargs), "vecenv_type": "RLGPU"}
)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.use_global_obs = self.env.num_states > 0

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"], infos = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, infos
        else:
            return self.full_state["obs"], infos

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space
        info["amp_observation_space"] = self.env.amp_observation_space
        info["enc_amp_observation_space"] = self.env.enc_amp_observation_space

        if hasattr(self.env.task, "get_task_obs_size"):
            info["task_obs_size"] = self.env.task.get_task_obs_size()
        else:
            info["task_obs_size"] = 0

        if self.use_global_obs:
            info["state_space"] = self.env.state_space
            print(info["action_space"], info["observation_space"], info["state_space"])
        else:
            print(info["action_space"], info["observation_space"])

        return info


vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))


def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder("amp", lambda **kwargs: amp_agent.AMPAgent(**kwargs))
    runner.player_factory.register_builder("amp", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder(
        "amp", lambda network, **kwargs: amp_models.ModelAMPContinuous(network)
    )
    runner.model_builder.network_factory.register_builder("amp", lambda **kwargs: amp_network_builder.AMPBuilder())

    runner.algo_factory.register_builder("parkour", lambda **kwargs: parkour_agent.ParkourAgent(**kwargs))
    # runner.player_factory.register_builder("parkour", lambda **kwargs: parkour_players.？(**kwargs))
    runner.model_builder.model_factory.register_builder(
        "parkour", lambda network, **kwargs: amp_models.ModelAMPContinuous(network)
    )
    runner.model_builder.network_factory.register_builder(
        "parkour", lambda **kwargs: parkour_network_builder.ParkourBuilder(**kwargs)
    )

    return runner


@hydra.main(version_base="1.3", config_path="./cfg", config_name="train.yaml")
def main(cfg: DictConfig):
    global cfg_complete
    utils.config.parse_config(cfg)
    set_np_formatting()
    setproctitle(cfg["config"]["task_name"])
    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    # if args.motion_file:
    #     cfg["env"]["motion_file"] = args.motion_file

    # Create default directories for weights and statistics
    # cfg_train["params"]["config"]["train_dir"] = args.output_path

    # time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{args.wandb_run_name}_{time_str}"
    # assert args.track and wandb or not args.track, "Tracking requires wandb to be installed."
    # if cfg.track:
    #     wandb.init(
    #         project=args.wandb_project_name,
    #         sync_tensorboard=True,
    #         config=args,
    #         monitor_gym=True,
    #         save_code=True,
    #         name=run_name,
    #     )

    cfg_complete = OmegaConf.to_container(cfg, resolve=True)
    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load({"params": cfg_complete})
    runner.reset()
    runner.run(cfg)

    return


if __name__ == "__main__":
    main()
