#!/usr/bin/env python3

import argparse
import random

import numpy as np
import os
from pathlib import Path

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.imitation_learning.algorithm.bc_trainer import BCTrainer
#from habitat_baselines.imitation_learning.algorithm.runner_script import runner_function
from habitat import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="train a policy and validate it's performance on expert trajectories\
        (train) or evaluate the trained policy in an environment(eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--checkpoint-number", 
        type=int,
        required=False,
        default=-1,
        help="checkpoint number required when you are running eval, default: loads the final trained model"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, checkpoint_number:int = -1, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    #add sensors
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK_CONFIG.TASK.SENSORS.append("HEADING_SENSOR")
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_SENSOR")
    config.freeze()

    trainer = BCTrainer(config)
        
    if run_type == "train":
        trainer.train()
        return

    elif run_type == "eval":
        trainer.eval_checkpoint(checkpoint_number)
    
if __name__ == "__main__":
    main()
