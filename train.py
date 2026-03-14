from omegaconf import DictConfig, OmegaConf
from tqdm import trange

import torch
import hydra
import logging
import wandb
import os
import json

import numpy as np

import src.samplers
import src.model
from typing import Optional
from copy import deepcopy

from src.data_gen import TrainEvalDataset
from src.utils import set_random_seed, init_model, prepare_verify_probes_list
from src.grad_tracking import GradTracker, get_grad_tracker
from src.training import train
import src.training


# Register the custom resolver
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)


logging.basicConfig(level=logging.DEBUG, 
                    format='%(levelname)s %(asctime)s - %(message)s', 
                    datefmt='%d-%m-%Y %H:%M')


def config_asserts(cfg: DictConfig):

    if cfg["probe_interval"] is not None:
        assert cfg["num_steps"] % cfg["probe_interval"] == 0, "num_steps must be divisible by probe_interval"
    else:
        cfg["probe_interval"] = cfg["num_steps"] + 1

    if cfg["eval_interval"] is None:
        cfg["eval_interval"] = cfg["num_steps"] + 1

    #if cfg["eval_interval"] is not None:
    #    assert cfg["num_steps"] % cfg["eval_interval"] == 0, "num_steps must be divisible by eval_interval"
    #else:
    #    cfg["eval_interval"] = cfg["num_steps"] + 1


def one_training_run(cfg: DictConfig, dataset: TrainEvalDataset, grad_tracker: Optional[GradTracker] = None, will_grad_track: bool = False):
    grad_tracking_on = grad_tracker is not None

    if cfg["wandb"] and not will_grad_track:
        wandb.init(
            project="INSERT YOURS",
            entity="INSERT YOURS",
            name=cfg["wandb_name"],
            config=cfg,
        )
        wandb.config.update({"grad_tracking_on": grad_tracking_on})
    elif cfg["wandb"] and will_grad_track:
        logging.info("wandb is on but grad tracking (will be) on, so not initializing wandb")

    set_random_seed(cfg["seed"])

    model = init_model(cfg, dataset.sampler, will_grad_track or grad_tracking_on)

    # the main training loop (code in training.py)
    model, probes = train(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(ignore_index=-100),
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["lr"]),
        dataset=dataset,
        cfg=cfg,
        grad_tracker=grad_tracker,
    )

    if cfg["model_save_path"] is not None:
        torch.save(model.state_dict(), cfg["model_save_path"])

    if wandb.run:
        wandb.finish()

    grad_tracker = get_grad_tracker(model, probes, dataset.sampler, cfg)

    return grad_tracker


def instantiate_dataset(cfg: DictConfig):
    sampler = hydra.utils.instantiate(cfg["world"])
    if cfg["dataset_path"] is not None:

        identifier = deepcopy(cfg["world"])
        identifier["train_dataset_size"] = cfg["train_dataset_size"]
        identifier["eval_dataset_size"] = cfg["eval_dataset_size"]
        identifier["deduplication_size"] = cfg["deduplication_size"]
        identifier = str(identifier)

        return TrainEvalDataset.load_or_save(
            dir=cfg["dataset_path"],
            sampler=sampler,
            identifier=identifier,
            train_size=cfg["train_dataset_size"],
            eval_size=cfg["eval_dataset_size"],
            deduplication_size=cfg["deduplication_size"],
            tqdm_on=True,
            sharding_size=cfg.get("sharding_size")
        )
    else:
        # If sharding is requested without dataset_path, disallow to keep a single source of truth
        if cfg.get("sharding_size") is not None:
            raise ValueError("cfg.sharding_size is set but cfg.dataset_path is None. Set dataset_path to the shard directory.")
        return TrainEvalDataset(
            sampler=sampler,
            train_size=cfg["train_dataset_size"],
            eval_size=cfg["eval_dataset_size"],
            deduplication_size=cfg["deduplication_size"]
        )


# this is needed so that hydra can automatically construct a config dictionary and pass it to main
@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f"Running with config: {cfg}")

    config_asserts(cfg)

    dataset = instantiate_dataset(cfg)

    if cfg["only_save_dataset"]:
        logging.info(f"Only saving dataset, exiting")
        return

    will_grad_track = prepare_verify_probes_list(cfg)

    logging.info(f"Starting the first run!")
    grad_tracker = one_training_run(cfg, dataset, grad_tracker=None, will_grad_track=will_grad_track)

    if will_grad_track:
        logging.info(f"Starting the second run!")
        grad_tracker = one_training_run(cfg, dataset, grad_tracker=grad_tracker)

if __name__ == "__main__":
    main()
