# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch

SIM_TIMESTEP = 1.0 / 60.0


def set_np_formatting():
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]"
    )


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def parse_phys_eigen(physics_engine):
    if physics_engine == "flex":
        return gymapi.SIM_FLEX
    elif physics_engine == "physx":
        return gymapi.SIM_PHYSX
    else:
        raise ValueError("Invalid physics engine: {}".format(physics_engine))


def parse_sim_params(cfg):
    # initialize sim
    args = cfg["config"]
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = args["slices"]

    if args["physics_engine"] == "flex":
        if args["device"] != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args["physics_engine"] == "physx":
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args["use_gpu"]
        sim_params.physx.num_subscenes = args["subscenes"]
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args["use_gpu"]
    sim_params.physx.use_gpu = args["use_gpu"]

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg["environment"]:
        gymutil.parse_sim_config(cfg["environment"]["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args["physics_engine"] == "physx" and args["num_threads"] > 0:
        sim_params.physx.num_threads = args["num_threads"]

    return sim_params


from isaacgym import gymapi
from omegaconf import DictConfig, OmegaConf, SCMode


def parse_config(cfg):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    rank = 0
    if cfg.config.multi_gpu:
        import horovod.torch as hvd

        hvd.init()
        rank = hvd.rank()
        print("Horovod rank: ", rank)
    cfg["seed"] = cfg["seed"] + rank
    cfg["device_type"] = "cuda"
    cfg["rank"] = rank
    cfg["rl_device"] = "cuda:" + str(rank)

    cfg["name"] = cfg["environment"]["task"]
    cfg["headless"] = cfg["config"]["headless"]
    cfg["play"] = not cfg["train"]
    cfg["config"]["num_actors"] = cfg["environment"]["env"]["numEnvs"]

    # Set physics domain randomization
    # if "task" in cfg:
    #     if "randomize" not in cfg["task"]:
    #         cfg["task"]["randomize"] = args.randomize
    #     else:
    #         cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    # else:
    #     cfg["task"] = {"randomize": False}
    #################3
    exp_name = cfg["config"]["name"]

    # if args.experiment != "Base":
    #     if args.metadata:
    #         exp_name = "{}_{}_{}_{}".format(
    #             args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1]
    #         )

    #         if cfg["task"]["randomize"]:
    #             exp_name += "_DR"
    #     else:
    #         exp_name = args.experiment

    # if benchmark:
    #     custom_parameters += [
    #         {"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
    #         {
    #             "name": "--random_actions",
    #             "action": "store_true",
    #             "help": "Run benchmark with random actions instead of inferencing",
    #         },
    #         {"name": "--bench_len", "type": int, "default": 10, "help": "Number of timing reports"},
    #         {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"},
    #     ]
