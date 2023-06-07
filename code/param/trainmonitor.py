import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import base58
import hivemind
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from hivemind import get_dht_time
from hivemind.dht.schema import (BytesWithPublicKey, RSASignatureValidator,
                                 SchemaValidator)
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

import shared.utils as utils
from shared.utils import write_conf_to_file
import wandb
from shared.progress_parser import (GlobalTrainingProgress,
                                    LocalTrainingProgress,
                                    TrainingProgressSchema,
                                    parse_progress_metadata)

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


class PeerIDTranslator:
    """ """

    def __init__(self):
        """ """
        self.id_to_num = dict()

    def add_id(self, id_str):
        """ """
        if not (id_str in self.id_to_num.keys()):
            self.id_to_num[id_str] = max(self.id_to_num.values(), default=0) + 1
            logger.info(
                f"PeerIDTranslator.add_id: Registered peer id {id_str} to value: {self.id_to_num[id_str]}"
            )

    def translate(self, id):
        """ """
        id_str = base58.b58encode(id).decode()
        self.add_id(id_str)
        return self.id_to_num[id_str]


class TrainMonitor:
    def __init__(self, target_batch_size):
        self.last_global_training_progress = None
        self.target_batch_size = target_batch_size
        self.undershot_TBS = 0
        self.reached_TBS = 0
        self.overshot_TBS = 0
        self.actual_training_samples = 0
        self.last_epoch_timestamp_s = 0

    def log_epoch_change(self, global_training_progress):
        if self.last_global_training_progress == None:
            self.last_global_training_progress = global_training_progress
            self.last_epoch_timestamp_s = time.perf_counter()
            return
        # if epoch changed check accumulated samples
        if (
            self.last_global_training_progress.global_epoch
            != global_training_progress.global_epoch
        ):
            current_time_s = time.perf_counter()
            epoch_s = current_time_s - self.last_epoch_timestamp_s
            self.last_epoch_timestamp_s = current_time_s
            epoch_samples = self.last_global_training_progress.samples_accumulated
            if epoch_samples == 0:
                epoch_sps = 0
            else:
                epoch_sps = epoch_samples / epoch_s

            self.actual_training_samples += epoch_samples
            if epoch_samples < self.target_batch_size:
                self.undershot_TBS += 1
            elif epoch_samples > self.target_batch_size:
                self.overshot_TBS += 1
            else:
                self.reached_TBS += 1

            prefix = "03_hivemind/"
            data = {
                f"{prefix}undershot_TBS": self.undershot_TBS,
                f"{prefix}reached_TBS": self.reached_TBS,
                f"{prefix}overshot_TBS": self.overshot_TBS,
                f"{prefix}total_training_samples": self.actual_training_samples,
                f"{prefix}per_epoch_training_samples": epoch_samples,
                f"{prefix}epoch_s": epoch_s,
                f"{prefix}epoch_based_sps": epoch_sps,
            }
            #   logger.info(f"\n{pd.DataFrame(data, index=[0])}")
            wandb.log(data, commit=False)

        self.last_global_training_progress = global_training_progress


def log_local_training_progress(
    peer_id_translator, local_training_progress_list, global_progress
):
    """ """
    prefix = "04_peers/"
    for peer in local_training_progress_list:
        unique_peer_num = peer_id_translator.translate(peer.peer_id)
        peer_name = f"peer_{unique_peer_num}"
        peer_data = {
            f"{prefix + peer_name}_epoch": peer.epoch,
            f"{prefix + peer_name}_samples_accumulated": peer.samples_accumulated,
            f"{prefix + peer_name}_samples_per_second": round(
                peer.samples_per_second, 2
            ),
            # f"{prefix + peer_name}_client_mode": peer.client_mode,
            f"{prefix + peer_name}_epoch_oos": (
                global_progress.global_epoch - peer.epoch
            ),
        }
        logger.info(f"{pd.DataFrame(peer_data, index=[0]).to_string()}")
        wandb.log(peer_data, commit=False)


def log_global_training_progress(global_progress):
    """ """
    prefix = "03_hivemind/"
    data = {
        f"{prefix}global_epoch": global_progress.global_epoch,
        f"{prefix}global_samples_accumulated": global_progress.samples_accumulated,
        f"{prefix}actual_total_samples_per_second": global_progress.actual_total_samples_per_second,
        f"{prefix}optimistic_total_samples_per_second": global_progress.optimistic_total_samples_per_second,
        f"{prefix}estimated_current_samples": global_progress.estimated_current_samples,
        f"{prefix}num_peers": global_progress.num_peers,
        f"{prefix}num_clients": global_progress.num_clients,
        f"{prefix}eta_next_epoch_s": global_progress.eta_next_epoch_s,
        f"{prefix}next_fetch_time_s": global_progress.next_fetch_time_s,
    }
    logger.info(f"{pd.DataFrame(data, index=[0]).to_string()}")
    wandb.log(data)


def run_monitor(config):
    """ """
    dht = init_dht(
        initial_peers=config.get("initial_peers", "").split(","),
        run_name=config["run_name"],
    )

    peer_id_translator = PeerIDTranslator()
    trainmonitor = TrainMonitor(target_batch_size=config["target_batch_size"])

    state_averaging_key = f"{config['run_name']}_state_averager.all_averagers"
    grad_averaging_key = f"{config['run_name']}_grad_averager.all_averagers"
    training_progress_key = f"{config['run_name']}_progress"
    # main loop
    while True:
        try:
            metadata, expiration = dht.get(
                key=training_progress_key, return_future=False, latest=True
            )
            global_progress, local_training_progress_list = parse_progress_metadata(
                metadata=metadata, target_batch_size=config["target_batch_size"]
            )
            log_global_training_progress(global_progress=global_progress)
            trainmonitor.log_epoch_change(global_training_progress=global_progress)
            log_local_training_progress(
                peer_id_translator=peer_id_translator,
                local_training_progress_list=local_training_progress_list,
                global_progress=global_progress,
            )
            time.sleep(config["monitor_log_frequency_ms"] / 1000)
        except BaseException as e:
            logger.info(f"Exception: {e}")
            # typically at the start because the monitor is too fast to start up
            time.sleep(1)

    return 0


def init_dht(initial_peers: str, run_name: str):
    """ """
    trials = 0
    MAX_TRIALS = 5
    PORT = 45555
    while trials < MAX_TRIALS:
        try:
            # Create DHT: a decentralized key-value storage shared between peers
            dht = hivemind.DHT(
                start=True,
                host_maddrs=[
                    f"/ip4/0.0.0.0/tcp/{PORT}",
                    f"/ip4/0.0.0.0/udp/{PORT}/quic",
                ],
                initial_peers=initial_peers,
                use_relay=True,
                use_auto_relay=True,
                client_mode=True,
            )
            break
        except hivemind.p2p.P2PDaemonError as e:
            logger.error(e.args)
            trials += 1

        if trials >= MAX_TRIALS:  # too many trials, kill this one
            raise ValueError(
                "init_dht: Failed to load DHT state from other peers after 5 trials"
            )
        # wait for random time, so another peer can jump in
        time.sleep(random.randint(1, 5))

    # visible_maddrs = [str(maddr) for maddr in dht.get_visible_maddrs()]
    signature_validator = RSASignatureValidator(None)
    local_public_key = signature_validator.local_public_key
    dht.add_validators(
        [
            SchemaValidator(TrainingProgressSchema, prefix=run_name),
            signature_validator,
        ]
    )
    return dht


def parse_args():
    """ """
    parser = argparse.ArgumentParser(description="Auxiliary")

    parser.add_argument(
        "--run_name",
        help="Name of current run",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host where the machine is currently running on",
    )
    parser.add_argument(
        "--initial_peers",
        type=str,
        help="Initial DHT peers",
        default="",
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--monitor_log_frequency_ms", type=int, default=1000)
    parser.add_argument("--max_hivemind_epochs", type=int, default=-1)
    parser.add_argument("--target_batch_size", type=int)
    parser.add_argument("--matchmaking_time", type=float)
    parser.add_argument("--averaging_timeout", type=float)
    args, _ = parser.parse_known_args()
    return vars(args)


def init_wandb(args: dict) -> dict:
    """ """

    config = {"type": "trainmonitor", **args}
    name = f"trainmonitor-{config['run_name']}"
    wandb.init(
        config=config,
        name=name,
    )
    logger.info(config)
    utils.write_conf_to_file(config)
    return config


def main() -> int:
    """ """
    load_dotenv(override=True)
    args = parse_args()
    config = init_wandb(args)
    return run_monitor(config)


if __name__ == "__main__":
    sys.exit(main())
