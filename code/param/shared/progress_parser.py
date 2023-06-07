import argparse
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
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
from hivemind.optim.progress_tracker import (GlobalTrainingProgress,
                                             LocalTrainingProgress,
                                             TrainingProgressSchema)
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from pydantic import BaseModel, StrictBool, StrictFloat, confloat, conint

import shared.utils as utils
import wandb

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


@dataclass(frozen=True)
class GlobalTrainingProgressV2:
    global_epoch: int
    samples_accumulated: int
    actual_total_samples_per_second: float
    optimistic_total_samples_per_second: float
    estimated_current_samples: int
    target_batch_size: int
    num_peers: int
    num_clients: int
    eta_next_epoch_s: float
    next_fetch_time_s: float

    def to_pd(self):
        """ """
        return pd.DataFrame(self.to_dict(), index=[0])

    def to_dict(self):
        """ """
        return asdict(self)


def parse_progress_metadata(
    metadata,
    target_batch_size: int,
    expected_drift_peers: float = 3,
    expected_drift_rate: float = 0.2,
    min_refresh_period: float = 0.5,
    max_refresh_period: float = 10,
):
    """Process the LocalTrainingProgress from each peer to gather
    the global state of the training progress.

    This method is mostly a copy of the "_parse_swarm_progress_data" method
    of the hivemind/optim/progress_tracker.py with local training information set to zero.

    Some facts:
    1. The global epoch is defined by the highest local epoch.
    2. Only peers with the matching global epoch are counted to the TBS
    3. All peers (even staggering ones) are counted to the samples/s

    :param metadata: data gathered from the DHT by the "{run_id}_progress" key
    :param target_batch_size: provided by the run-starter
    :param expected_drift_peers: some default, unknown why
    :param expected_drift_rate: some default, unknown why
    :param min_refresh_period: minimum time after each refresh DHT get for peers, not used by us but tracked
     to estimate how often they access the DHT
    :param max_refresh_period: maximum time after each refresh for DHT peers get
    :return GlobalTrainingProgressV2, [LocalTrainingProgress]
    """
    current_time = get_dht_time()

    # only accept a dict in the form of LocalTrainingProgress
    valid_peer_entries = [
        LocalTrainingProgress.parse_obj(peer_state.value)
        for peer_state in metadata.values()
        if peer_state.value is not None
    ]

    # some peers are not doing training, just helping with averaging called client_mode
    num_peers = len(valid_peer_entries)
    num_clients = sum(peer.client_mode for peer in valid_peer_entries)

    # logger.info(f"num_peers: {num_peers}")

    # epoch is defined by the highest epoch from all peers
    global_epoch = 0  # monitor does not have its own epoch
    for peer in valid_peer_entries:
        if not peer.client_mode:
            global_epoch = max(global_epoch, peer.epoch)
    #       logger.info(
    #           f"- peer: {base58.b58encode(peer.peer_id).decode()}, epoch: {peer.epoch}, sps: {peer.samples_per_second}"
    #       )
    # monitor does not have a samples/s or accumulate samples
    total_samples_accumulated = 0
    estimated_current_samples = 0
    # by hivemind's definition this speed ignores invalid peers assuming they will join sooner or later
    total_samples_per_second = 0
    # we add actual speed without invalid peers
    actual_total_samples_per_second = 0

    for peer in valid_peer_entries:
        total_samples_per_second += peer.samples_per_second
        if peer.epoch == global_epoch:
            actual_total_samples_per_second += peer.samples_per_second
            total_samples_accumulated += peer.samples_accumulated
            peers_currently_not_reported_progress = (
                max(0.0, current_time - peer.time) * peer.samples_per_second
            )
            estimated_current_samples += (
                peer.samples_accumulated + peers_currently_not_reported_progress
            )

    estimated_samples_remaining = target_batch_size - estimated_current_samples
    estimated_time_to_next_epoch = (
        max(0, estimated_samples_remaining) / total_samples_per_second
    )

    expected_max_peers = max(
        num_peers + expected_drift_peers, num_peers * (1 + expected_drift_rate)
    )
    time_to_next_fetch = float(
        np.clip(
            a=estimated_time_to_next_epoch * num_peers / expected_max_peers,
            a_min=min_refresh_period,
            a_max=max_refresh_period,
        )
    )
    global_training_progress = GlobalTrainingProgressV2(
        global_epoch=global_epoch,
        samples_accumulated=total_samples_accumulated,
        actual_total_samples_per_second=actual_total_samples_per_second,
        optimistic_total_samples_per_second=total_samples_per_second,
        estimated_current_samples=round(estimated_current_samples, 0),
        target_batch_size=target_batch_size,
        num_peers=num_peers,
        num_clients=num_clients,
        eta_next_epoch_s=round(estimated_time_to_next_epoch, 2),
        next_fetch_time_s=round(time_to_next_fetch, 2),
    )
    return global_training_progress, valid_peer_entries
