import argparse
import ast
import json
import os
import random
import sys
import time
import uuid
import pickle
import webdataset
from typing import Callable, List, Tuple, Union

from dotenv import load_dotenv

# load as soon as possible because of "file_descriptor"
load_dotenv(override=True)

import hivemind
import torch
from torch.utils.data import Subset
import torch.nn.functional as F
import torchvision
from hivemind.dht.schema import (BytesWithPublicKey, RSASignatureValidator,
                                 SchemaValidator)
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

import shared.utils as utils
from shared.utils import write_conf_to_file
import wandb
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

    wandb.define_metric("01_general/val_accuracy", summary="max")

    visible_maddrs: List[str] = []
    dht: hivemind.DHT = None

    if config.get("run_hivemind"):
        dht, visible_maddrs = setup_hivemind_dht(
            config.get("initial_peers", "").split(","), run_name=config.get("run_name")
        )

    wandb.config.update({"visible_maddrs": visible_maddrs})
    write_conf_to_file(wandb.config)

    logger.info("Loading dataset...")
    trainset, valset, num_classes, input_channels, accuracy_fn = select_dataset(
        config.get("dataset"),
        config.get("data_folder"),
        batch_size=config.get("batch_size_per_step"),
    )

    # best practice from https://webdataset.github.io/webdataset/sharding/
    if isinstance(trainset, webdataset.WebDataset):
        train_dataloader = torch.utils.DataLoader(
            trainset.batched(config.get("batch_size_per_step")), batch_size=None, num_workers=4
        )
        val_dataloader = torch.utils.DataLoader(
            valset.batched(config.get("batch_size_per_step")), batch_size=None, num_workers=4
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            trainset, shuffle=False, batch_size=config.get("batch_size_per_step"), num_workers=4
        )
        val_dataloader = torch.utils.data.DataLoader(
            valset, shuffle=False, batch_size=config.get("batch_size_per_step"), num_workers=4
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
            opt,
            scheduler,
            grad_compression,
            state_averaging_compression,
            dht,
            config.get("run_name"),
            config.get("gradient_accumulation_steps"),
            config.get("batch_size_per_step"),
            config.get("target_batch_size"),
            config.get("use_local_updates"),
            config.get("matchmaking_time"),
            config.get("averaging_timeout"),
            config.get("is_first_peer"),
            config.get("delay_optimizer_step"),
            config.get("delay_state_averaging"),
            config.get("average_state_every"),
        )

    try:
        start_training(
            train_dataloader,
            val_dataloader,
            model,
            opt,
            accuracy_fn,
            dht,
            config.get("epochs"),
            config.get("gradient_accumulation_steps"),
            config.get("max_steps"),
            config.get("max_hivemind_epochs"),
            config.get("batch_size_per_step"),
            config.get("log_frequency"),
            config.get("loss_upper_threshold"),
            config.get("target_batch_size"),
            config.get("run_name"),
        )
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt or SIGINT, shutting down")
        shutdown(opt)
        wandb.finish(exit_code=0)


class EpochMonitor:
    def __init__(self, dht, target_batch_size, run_name):
        self.last_global_training_progress = None
        self.actual_training_samples = 0
        self.dht = dht
        self.target_batch_size = target_batch_size
        self.training_progress_key = f"{run_name}_progress"

    def epoch_changed(self):
        """ """
        # fetch the global trainig progress from the dht
        metadata, _ = self.dht.get(
            key=self.training_progress_key, return_future=False
        )  # , latest=True)
        global_training_progress, _ = parse_progress_metadata(
            metadata=metadata, target_batch_size=self.target_batch_size
        )

        result = False
        # first call to function, set the initial progress
        if self.last_global_training_progress == None:
            self.last_global_training_progress = global_training_progress

        # if epoch changed add the last accumulated samples to global counter
        if (
            self.last_global_training_progress.global_epoch
            != global_training_progress.global_epoch
        ):
            epoch_samples = self.last_global_training_progress.samples_accumulated
            self.actual_training_samples += epoch_samples
            result = True

        # reset to new progress state
        self.last_global_training_progress = global_training_progress
        return result


def reached_max_hivemind_epochs(
    opt: Union[torch.optim.Optimizer, hivemind.Optimizer], max_hivemind_epochs: int
):
    if isinstance(opt, hivemind.Optimizer):
        return opt.tracker.global_epoch >= max_hivemind_epochs
    else:
        return False


def start_training(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    opt: Union[torch.optim.Optimizer, hivemind.Optimizer],
    accuracy_fn: Callable,
    dht: hivemind.DHT,
    epochs: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    max_hivemind_epochs: int,
    batch_size_per_step: int,
    log_frequency: int,
    loss_upper_threshold: float,
    target_batch_size: int,
    run_name: str,
):

    epoch_monitor = EpochMonitor(
        dht=dht, target_batch_size=target_batch_size, run_name=run_name
    )
    time.sleep(20)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def perform_validation():
        """ """
        with wandbtimeit("val_dataset_eval"):
            val_accuracy = utils.test_model(
                model=model, dataloader=val_dataloader, device=device
            )
        logger.info(
            f"Hivemind epoch {epoch_monitor.last_global_training_progress.global_epoch - 1}"
            + f" validation accuracy: {val_accuracy}"
        )
        wandb.log({"01_general/val_accuracy": val_accuracy})

    @torch.no_grad()
    def backup_state(path):

        pickle.dumps(
            { "model": model.state_dict(),
              "optimizer": opt.state_dict()
            },
            path
        )

    relative_model_path = "/home/ubuntu/rbgstorage/temp/isenko/models/{run_name}"
    while True:
        if epoch_monitor.epoch_changed():
            opt.load_state_from_peers()
            perform_validation()
            #backup_state(path = f"{relative_model_path}/{epoch_monitor.last_global_training_progress.global_epoch - 1}.pkl")
        else:
            time.sleep(1)


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
    total_batch_size_per_step = batch_size_per_step * gradient_accumulation_steps

    #    tracker_opts = {"min_refresh_period": 0.05}

    # Set up a decentralized optimizer that will average with peers in background
    opt = hivemind.Optimizer(
        dht=dht,  # use a DHT that is connected with other peers
        # unique identifier of this collaborative run
        run_id=run_name,
        # each call to opt.step adds this many samples towards the next epoch
        batch_size_per_step=total_batch_size_per_step,
        # after peers collectively process this many samples, average weights and begin the next epoch
        target_batch_size=target_batch_size,
        optimizer=opt,  # wrap the SGD optimizer defined above
        # perform optimizer steps with local gradients, average parameters in background
        use_local_updates=use_local_updates,
        # when averaging parameters, gather peers in background for up to this many seconds
        matchmaking_time=matchmaking_time,
        # give up on averaging if not successful in this many seconds
        averaging_timeout=averaging_timeout,
        grad_compression=grad_compression,
        state_averaging_compression=state_averaging_compression,
        #        delay_optimizer_step=delay_optimizer_step,
        delay_state_averaging=delay_state_averaging,
        average_state_every=average_state_every,
        scheduler=scheduler,
        verbose=False,  # print logs incessently
        #         tracker_opts=tracker_opts,  #
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
        "type": "lazybee",
        **vars(args),
    }

    config.update({"optimizer_params": json.loads(config.get("optimizer_params"))})
    config.update({"scheduler_params": json.loads(config.get("scheduler_params"))})

    name = f"lazybee-{args.host}-{args.run_name}"
    wandb.init(
        config=config,
        name=name,
    )

    start_bee(config)
