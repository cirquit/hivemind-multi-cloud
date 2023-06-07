import argparse
import ast
import json
import os
import random
import sys
import time
import uuid
import pickle
from typing import Callable, List, Tuple, Union
from dotenv import load_dotenv

# load as soon as possible because of "file_descriptor"
load_dotenv(override=True)

import hivemind
import torch
import wandb
from torch.utils.data import Subset
import torch.nn.functional as F
import torchvision
from hivemind.dht.schema import (BytesWithPublicKey, RSASignatureValidator,
                                 SchemaValidator)
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

import shared.utils as utils
from shared.utils import write_conf_to_file

from datasets import select_dataset
from grad_compression import select_grad_compression
from models import select_model
from shared.monitor import Monitor
from optimizers import select_optimizer
from schedulers import select_scheduler
from shared.progress_parser import (GlobalTrainingProgress,
                                    LocalTrainingProgress,
                                    TrainingProgressSchema,
                                    parse_progress_metadata)
from shared.timers import WandbTimeIt as wandbtimeit

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)
mon = Monitor()

def shutdown(opt: Union[torch.optim.Optimizer, hivemind.Optimizer]):
    if isinstance(opt, hivemind.Optimizer):
        opt.shutdown()
        opt.dht.shutdown()
        logger.info("Waiting for DHT to terminate...")
        opt.dht.join(timeout=10)
        logger.info("DHT Terminated successfully")


def start_bee(config: dict):
    """Prepare the experimental setup, which includes
    - setup monitoring
    - drop the caches
    - fix the seed
    - prepare the dataset
    - prepare the model
    - prepare the optimizer + optional hivemind
    """

    utils.drop_io_cache()
    utils.disable_torch_debugging()
    if config.get("seed") > -1:
        utils.seed_everything(config.get("seed"))

    wandb.log(mon.get_static_info())

    visible_maddrs: List[str] = []
    dht: hivemind.DHT = None

    if config.get("run_hivemind"):
        dht, visible_maddrs = setup_hivemind_dht(
            config.get("initial_peers", "").split(","), run_name=config.get("run_name")
        )

    wandb.config.update({"visible_maddrs": visible_maddrs})
    write_conf_to_file(wandb.config)

    logger.info("Loading dataset...")
    _, _, num_classes, input_channels, _ = select_dataset(
        config.get("dataset"),
        config.get("data_folder"),
        batch_size=config.get("batch_size_per_step"),
        dont_load=True
    )

    model: torch.nn.Module = select_model(
        config.get("model"), num_classes, input_channels
    )

    opt = select_optimizer(
        config.get("optimizer"),
        config.get("optimizer_params"),
        model,
    )

    scheduler = None
    if config.get("scheduler") != None:
        scheduler = select_scheduler(
            config.get("scheduler"),
            config.get("scheduler_params"),
            opt,
        )

    if config.get("run_hivemind"):
        grad_compression = select_grad_compression(config.get("grad_compression"))
        state_averaging_compression = select_grad_compression(
            config.get("state_averaging_compression")
        )

        opt = setup_hivemind_training(
            opt=opt,
            scheduler=scheduler,
            grad_compression=grad_compression,
            state_averaging_compression=state_averaging_compression,
            dht=dht,
            run_name=config.get("run_name"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps"),
            batch_size_per_step=config.get("batch_size_per_step"),
            target_batch_size=config.get("target_batch_size"),
            use_local_updates=config.get("use_local_updates"),
            matchmaking_time=config.get("matchmaking_time"),
            averaging_timeout=config.get("averaging_timeout"),
            is_first_peer=config.get("is_first_peer"),
            delay_optimizer_step=config.get("delay_optimizer_step"),
            delay_state_averaging=config.get("delay_state_averaging"),
            average_state_every=config.get("average_state_every"),
        )

    try:
        start_training(
            opt=opt,
            dht=dht,
            max_hivemind_epochs=config.get("max_hivemind_epochs"),
            target_batch_size=config.get("target_batch_size")
        )
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt or SIGINT, shutting down")
    finally:
        shutdown(opt)
        wandb.finish(exit_code=0)


def reached_max_hivemind_epochs(
    opt: Union[torch.optim.Optimizer, hivemind.Optimizer], max_hivemind_epochs: int
):
    if isinstance(opt, hivemind.Optimizer):
        return opt.tracker.global_epoch >= max_hivemind_epochs


def start_training(
    opt: Union[torch.optim.Optimizer, hivemind.Optimizer],
    dht: hivemind.DHT,
    max_hivemind_epochs: int,
    target_batch_size: int
):

    # wait until enough peers showed up
    while True:
        try:
            # fetch the global trainig progress from the dht
            metadata, _ = dht.get(
                key=opt.tracker.training_progress_key, return_future=False, latest=True)
            global_training_progress, _ = parse_progress_metadata(
                metadata=metadata, target_batch_size=target_batch_size
            )
        except Exception as e:
            # if no state was logged, this will throw
            logger.info(f"Exception {e}")
            time.sleep(1)
        else:
            # if we have 2+ peers, we can try to help with averaging
            if global_training_progress.num_peers >= 2:
                break

    # help out with averaging
    while True:
        with wandbtimeit("aux_peer_step") as aux_peer_step_s:
            opt.step()

        wandb.log(
            {
                **mon.get_sys_info(),
                "02_timing/aux_step_s": aux_peer_step_s.delta_time_s
            }
        )

        # termination condition
        if reached_max_hivemind_epochs(opt=opt, max_hivemind_epochs=max_hivemind_epochs):
            break


def setup_hivemind_dht(initial_peers: List[str], run_name):
    trials = 0
    MAX_N_RETIRES = 24
    REFRESH_TIME_S = 5
    dht: hivemind.DHT = None

    while trials < MAX_N_RETIRES:
        try:
            # quick fix
            if initial_peers == [""]:
                initial_peers = []
            # Create DHT: a decentralized key-value storage shared between peers
            dht = hivemind.DHT(
                start=True,
                host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                initial_peers=initial_peers,
            )
            break
        except hivemind.p2p.P2PDaemonError as e:
            logger.error(e.args)
            trials += 1

        if trials >= MAX_N_RETIRES:  # too many trials, kill this one
            raise ValueError(
                f"Failed to load state from other peers after {MAX_N_RETIRES} trials"
            )
        # wait for random time, so another peer can jump in
        time.sleep(REFRESH_TIME_S)

    visible_maddrs = [str(maddr) for maddr in dht.get_visible_maddrs()]
    signature_validator = RSASignatureValidator(None)
    local_public_key = signature_validator.local_public_key
    dht.add_validators(
        [
            SchemaValidator(TrainingProgressSchema, prefix=run_name),
            signature_validator,
        ]
    )
    return dht, visible_maddrs


def setup_hivemind_training(
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    grad_compression: hivemind.CompressionBase,
    state_averaging_compression: hivemind.CompressionBase,
    dht: hivemind.DHT,
    run_name: str,
    gradient_accumulation_steps: int,
    batch_size_per_step: int,
    target_batch_size: int,
    use_local_updates: bool,
    matchmaking_time: float,
    averaging_timeout: float,
    is_first_peer: bool,
    delay_optimizer_step: bool,
    delay_state_averaging: bool,
    average_state_every: int,
) -> Tuple[List[str], hivemind.Optimizer]:

    #    tracker_opts = {"min_refresh_period": 0.05}

    # Set up a decentralized optimizer that will average with peers in background
    opt = hivemind.Optimizer(
        dht=dht,  # use a DHT that is connected with other peers
        # unique identifier of this collaborative run
        run_id=run_name,
        # after peers collectively process this many samples, average weights and begin the next epoch
        # auxiliary does not accumulate by themselves
        target_batch_size=target_batch_size,
        # wrap the  optimizer defined above
        optimizer=opt,
        # perform optimizer steps with local gradients, average parameters in background
        use_local_updates=use_local_updates,
        # when averaging parameters, gather peers in background for up to this many seconds
        matchmaking_time=matchmaking_time,
        # give up on averaging if not successful in this many seconds
        averaging_timeout=averaging_timeout,
        grad_compression=grad_compression,
        state_averaging_compression=state_averaging_compression,
        # delay_optimizer_step=delay_optimizer_step,
        delay_state_averaging=delay_state_averaging,
        average_state_every=average_state_every,
        scheduler=scheduler,
        auxiliary=True,
        # print logs incessently
        verbose=False,  
        # tracker_opts=tracker_opts,
    )

    if not is_first_peer:
        trials = 0
        MAX_TRIALS = 5
        while trials < MAX_TRIALS:
            try:
                opt.load_state_from_peers()
                break
            except BaseException as e:
                logger.error("Failed to obtain state from other peers, retrying...")
                logger.error(e.args)
                trials += 1

        if trials >= MAX_TRIALS:  # too many trials, kill this one
            raise ValueError("Failed to load state from other peers after 5 trials")

    return opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lazybee")

    parser.add_argument(
        "--run_name",
        help="Name of current run",
        default=str(uuid.uuid4()),
        required="WANDB_DISABLED" not in os.environ,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host where the machine is currently running on",
        required="WANDB_DISABLED" not in os.environ,
    )

    # general options for training
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--number_of_nodes", type=int, default=1)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument(
        "--optimizer_params",
        default="{}",
        help="Parameters to pass to the optimizer in a JSON format",
    )
    parser.add_argument("--scheduler", type=str, required=False, default=None)
    parser.add_argument("--scheduler_params", default="{}", help="Scheduler parameters")
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=1,
        help="After how many steps we should log",
    )
    parser.add_argument(
        "--batch_size_per_step", type=int, help="Batch size", required=True
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--loss_upper_threshold",
        type=float,
        default=10.0,
        help="Loss can safely grow up to this point. If above, a warning will be triggered. After three warnings, ",
    )
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--delay_optimizer_step", default=False, type=bool)

    # options for hivemind only
    parser.add_argument(
        "--run_hivemind",
        action="store_true",
        help="Whether this run will use hivemind or not",
    )
    parser.add_argument("--max_hivemind_epochs", type=int, default=-1)
    parser.add_argument(
        "--initial_peers",
        type=str,
        default="",
        help="Initial DHT peers",
    )
    parser.add_argument(
        "--is_first_peer",
        type=ast.literal_eval,
        default=False,
        help="Used to determine which node should lead the others",
        required="--run_hivemind" in sys.argv,
    )
    parser.add_argument(
        "--grad_compression", type=str, default="hivemind.NoCompression", required=False
    )
    parser.add_argument(
        "--state_averaging_compression",
        type=str,
        default="hivemind.NoCompression",
        required=False,
    )
    parser.add_argument(
        "--use_local_updates",
        type=ast.literal_eval,
        default=False,
        help="Whether the hivemind optimizer should use local updates",
        required="--run_hivemind" in sys.argv,
    )
    parser.add_argument(
        "--target_batch_size", type=int, required="--run_hivemind" in sys.argv
    )
    parser.add_argument(
        "--matchmaking_time", type=float, required="--run_hivemind" in sys.argv
    )
    parser.add_argument(
        "--averaging_timeout", type=float, required="--run_hivemind" in sys.argv
    )
    parser.add_argument(
        "--delay_state_averaging", type=bool, required="--run_hivemind" in sys.argv
    )
    parser.add_argument(
        "--average_state_every", type=int, required="--run_hivemind" in sys.argv
    )

    # don't throw if we are parsing unknown args
    args, unknown = parser.parse_known_args()

    config = {
        "type": "auxiliarybee",
        **vars(args),
    }

    config.update({"optimizer_params": json.loads(config.get("optimizer_params"))})
    config.update({"scheduler_params": json.loads(config.get("scheduler_params"))})

    name = f"auxiliarybee-{args.host}-{args.run_name}"
    wandb.init(
        config=config,
        name=name,
    )

    start_bee(config)
