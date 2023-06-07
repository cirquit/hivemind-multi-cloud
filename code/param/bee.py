import argparse
import ast
import json
import os
import random
import sys
import time
import uuid
from typing import Callable, List, Tuple, Union

import webdataset
from dotenv import load_dotenv

# load as soon as possible because of "file_descriptor"
load_dotenv(override=True)

import hivemind
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from wandb import AlertLevel

import shared.utils as utils
from datasets import select_dataloader, select_dataset
from grad_compression import select_grad_compression
from models import select_model
from optimizers import select_optimizer
from schedulers import select_scheduler
from shared.timers import WandbTimeIt as wandbtimeit
from shared.utils import shutdown_optimizer, start_monitor, write_conf_to_file

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def start_bee(config: dict):
    """Prepare the experimental setup
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

    wandb.define_metric("01_general/minibatch_loss", summary="min")

    visible_maddrs: List[str] = []
    dht: hivemind.DHT = None

    if config.get("run_hivemind"):
        dht, visible_maddrs = setup_hivemind_dht(
            initial_peers=config.get("initial_peers", "").split(","),
            announce_maddr_ip=config.get("announce_maddr_ip", None),
            announce_maddr_port=config.get("announce_maddr_port", None),
        )

    wandb.config.update({"visible_maddrs": visible_maddrs})
    write_conf_to_file(wandb.config)

    logger.info("Loading dataset...")
    trainset, valset, num_classes, input_channels, accuracy_fn = select_dataset(
        config.get("dataset"),
        config.get("data_folder"),
        batch_size=config.get("batch_size_per_step"),
    )

    train_dataloader = select_dataloader(
        dataset_name=config.get("dataset"),
        dataset=trainset,
        batch_size_per_step=config.get("batch_size_per_step"),
        num_workers=2,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model: torch.nn.Module = select_model(
        model_name=config.get("model"),
        num_classes=num_classes,
        input_channels=input_channels,
    ).to(device)

    opt = select_optimizer(
        config.get("optimizer"),
        config.get("optimizer_params"),
        model,
        lambda_return=not ("baseline" in config.get("run_name")),
    )

    scheduler = None
    if config.get("scheduler") != None:
        scheduler = select_scheduler(
            config.get("scheduler"),
            config.get("scheduler_params"),
            opt,
            lambda_return=not ("baseline" in config.get("run_name")),
        )

    if config.get("run_hivemind"):
        grad_compression = select_grad_compression(config.get("grad_compression"))
        state_averaging_compression = select_grad_compression(
            config.get("state_averaging_compression")
        )

        opt = setup_hivemind_training(
            model=model,
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
            model_name=config.get("model"),
            train_dataloader=train_dataloader,
            model=model,
            opt=opt,
            scheduler=scheduler,
            epochs=config.get("epochs"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps"),
            max_hivemind_epochs=config.get("max_hivemind_epochs"),
            max_runtime_s=config.get("max_runtime_s"),
            batch_size_per_step=config.get("batch_size_per_step"),
            pre_start_wait_time_s=config.get("pre_start_wait_time_s"),
            use_mixed_precision_fp16=config.get("use_mixed_precision_fp16"),
        )
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt or SIGINT, shutting down")
        shutdown_optimizer(opt)
        wandb.finish(exit_code=0)


def terminate_training(
    opt, max_hivemind_epochs, epoch, epochs, start_time_s, max_runtime_s
):
    """ """
    if (time.perf_counter() - start_time_s) >= max_runtime_s:
        logger.info(f"Terminating due to max_runtime_s of {max_runtime_s}.")
        wandb.alert(
            title="Run finished",
            text=f"Run finished: Reached maximum runtime {max_runtime_s}s",
            level=AlertLevel.WARN,
        )
        return True
    elif isinstance(opt, hivemind.Optimizer):
        if opt.tracker.global_epoch >= max_hivemind_epochs:
            # wait for the trainmonitor to get the new samples/s
            time.sleep(5)
            logger.info(
                f"Terminating due to maximum hivemind epoch {max_hivemind_epochs}."
            )
            wandb.alert(
                title="Run finished",
                text=f"Run finished: Reached maximum hivemind epochs {max_hivemind_epochs}",
                level=AlertLevel.WARN,
            )
        return opt.tracker.global_epoch >= max_hivemind_epochs
    else:
        if epoch >= epochs:
            logger.info(f"Terminating due to maximum epochs {epochs}.")
            wandb.alert(
                title="Run finished",
                text=f"Run finished: Reached maximum epochs {epochs}",
                level=AlertLevel.WARN,
            )
        return epoch >= epochs


def enable_dataparallel_training(model):
    """Enable dataparallel training if possible (same model on multiple GPUs)"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model.to(device)
    return model, device


def start_training(
    model_name: str,
    train_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    opt: Union[torch.optim.Optimizer, hivemind.Optimizer],
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    gradient_accumulation_steps: int,
    max_hivemind_epochs: int,
    max_runtime_s: int,
    batch_size_per_step: int,
    pre_start_wait_time_s: int,
    use_mixed_precision_fp16: bool,
):
    locally_processed_samples = 0
    amp_dtype = torch.float16 if use_mixed_precision_fp16 else torch.float32
    using_hivemind = isinstance(opt, hivemind.Optimizer)

    model, device = enable_dataparallel_training(model=model)

    time.sleep(pre_start_wait_time_s)
    start_time_s = time.perf_counter()

    roberta_training = "roberta_mlm" in model_name

    # terminate training on epoch OR max_hivemind_epoch OR max_runtime_s
    epoch = -1
    while True:
        if terminate_training(
            opt, max_hivemind_epochs, epoch, epochs, start_time_s, max_runtime_s
        ):
            shutdown_optimizer(opt)
            break

        # init starting variables
        step_time_start = time.perf_counter()
        epoch += 1

        for step, data in enumerate(train_dataloader):
            dataload_time_s = time.perf_counter() - step_time_start

            if roberta_training:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                labels = data["labels"]
            else:
                inputs = data[0]
                labels = data[1]

            locally_processed_samples += len(labels)
            logger.info(f">> Starting Step {step} <<")
            if (step % gradient_accumulation_steps) == 0:
                opt.zero_grad()
            with wandbtimeit("dataload_cuda_move") as dl_cuda_timer:
                if roberta_training:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                with wandbtimeit("forward") as forward_timer_s:
                    if roberta_training:
                        output = model(
                            input_ids=input_ids[0],
                            attention_mask=attention_mask[0],
                            labels=labels[0],
                        )
                    else:
                        output = model(inputs)
                with wandbtimeit("loss_calc") as loss_timer_s:
                    if roberta_training:
                        loss = output["loss"]
                    else:
                        loss = F.cross_entropy(output, labels)
                    if gradient_accumulation_steps > 1:
                        loss /= gradient_accumulation_steps
            with wandbtimeit("backward") as backward_timer_s:
                loss.backward()
            with wandbtimeit("opt_step") as opt_step_timer_s:
                if (step + 1) % gradient_accumulation_steps == 0:
                    opt.step()
                    # running baseline does not call LR_Scheduler, so we do it manually
                    if not using_hivemind:
                        scheduler.step()

            if using_hivemind:
                additional_log = {
                    "01_general/lr": opt.state_averager.scheduler.get_lr()[0],
                    "03_hivemind/global_epoch": opt.tracker.global_epoch,
                    "03_hivemind/num_peers": opt.tracker.global_progress.num_peers,
                }
            else:
                step_time_s = (
                    dataload_time_s
                    + forward_timer_s.delta_time_s()
                    + loss_timer_s.delta_time_s()
                    + backward_timer_s.delta_time_s()
                    + opt_step_timer_s.delta_time_s()
                    + dl_cuda_timer.delta_time_s()
                )
                additional_log = {
                    "01_general/lr": scheduler.get_lr()[0],
                    "01_general/step_based_sps": len(labels) / step_time_s,
                }

            wandb.log(
                {
                    **additional_log,
                    "01_general/minibatch_loss": (
                        loss * gradient_accumulation_steps
                    ).item(),
                    "01_general/step": step,
                    "01_general/dataset_iteration_count": epoch,
                    "01_general/locally_processed_samples": locally_processed_samples,
                    "02_timing/dataload_time_s": dataload_time_s,
                    "02_timing/step_time_s": dataload_time_s
                    + forward_timer_s.delta_time_s()
                    + loss_timer_s.delta_time_s()
                    + backward_timer_s.delta_time_s()
                    + opt_step_timer_s.delta_time_s()
                    + dl_cuda_timer.delta_time_s(),
                }
            )

            # restarting step time to measure dataloading time
            step_time_start = time.perf_counter()

            if terminate_training(
                opt, max_hivemind_epochs, epoch, epochs, start_time_s, max_runtime_s
            ):
                shutdown_optimizer(opt)
                break


def setup_hivemind_dht(
    initial_peers: List[str],
    announce_maddr_ip: str,
    announce_maddr_port: str,
):
    trials = 0
    MAX_N_RETIRES = 24
    REFRESH_TIME_S = 5
    PORT = 45555
    dht: hivemind.DHT = None
    kwargs = {}
    if announce_maddr_ip != None:
        announce_maddrs = f"/ip4/{announce_maddr_ip}/tcp/{announce_maddr_port}"
        kwargs["announce_maddrs"] = [announce_maddrs]

    while trials < MAX_N_RETIRES:
        try:
            # quick fix
            if initial_peers == [""]:
                initial_peers = []
            dht = hivemind.DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/{PORT}"],
                initial_peers=initial_peers,
                use_relay=True,
                use_auto_relay=True,
                client_mode=False,
                **kwargs,
            )
            break
        except hivemind.p2p.P2PDaemonError as e:
            logger.error(e.args)
            trials += 1

        if trials >= MAX_N_RETIRES:
            raise ValueError(
                f"Failed to load state from other peers after {MAX_N_RETIRES} trials"
            )
        time.sleep(REFRESH_TIME_S)

    visible_maddrs = [str(maddr) for maddr in dht.get_visible_maddrs()]
    return dht, visible_maddrs


def setup_hivemind_training(
    model,
    opt,  #: torch.optim.Optimizer or a lambda with param: opt(param)
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
    # tracker_opts = {"min_refresh_period": 0.05}
    # averager_opts = {"request_timeout": 1}

    # Set up a decentralized optimizer that will average with peers in background
    opt = hivemind.Optimizer(
        # needed to enable delayed parameter averaging
        params=model.parameters(),
        # use a DHT that is connected with other peers
        dht=dht,
        # unique identifier of this collaborative run
        run_id=run_name,
        # each call to opt.step adds this many samples towards the next epoch
        batch_size_per_step=total_batch_size_per_step,
        # after peers collectively process this many samples, average weights and begin the next epoch
        target_batch_size=target_batch_size,
        optimizer=opt,
        # perform optimizer steps with local gradients, average parameters in background
        use_local_updates=use_local_updates,
        # when averaging parameters, gather peers in background for up to this many seconds
        matchmaking_time=matchmaking_time,
        # give up on averaging if not successful in this many seconds
        averaging_timeout=averaging_timeout,
        grad_compression=grad_compression,
        state_averaging_compression=state_averaging_compression,
        delay_optimizer_step=delay_optimizer_step,
        delay_state_averaging=delay_state_averaging,
        average_state_every=average_state_every,
        scheduler=scheduler,
        # print logs incessently
        verbose=True,
        # tracker_opts=tracker_opts,
        # averager_opts=averager_opts
    )

    # if not is_first_peer:
    #    trials = 0
    #    MAX_TRIALS = 5
    #    while trials < MAX_TRIALS:
    #        try:
    #            #                opt.load_state_from_peers()
    #            print("not downloading state from peers")
    #            break
    #        except BaseException as e:
    #            logger.error("Failed to obtain state from other peers, retrying...")
    #            logger.error(e.args)
    #            trials += 1

    #    if trials >= MAX_TRIALS:  # too many trials, kill this one
    #        raise ValueError("Failed to load state from other peers after 5 trials")

    return opt


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Bee")
    parser.add_argument("--run_name")
    parser.add_argument("--host", default="127.0.0.1")

    # general options for training
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--number_of_nodes", type=int, default=1)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    # terminate on epoch OR max_runtime_s
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max_runtime_s", type=int)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--optimizer_params", default="{}", help="Optimizer parameters in JSON")
    parser.add_argument("--scheduler", type=str, required=False, default=None)
    parser.add_argument("--scheduler_params", default="{}", help="LR scheduler parameters in JSON")
    parser.add_argument("--batch_size_per_step", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--use_mixed_precision_fp16", type=ast.literal_eval, default=False)
    parser.add_argument("--monitor_log_frequency_ms", type=int, default=1, help="After how many ms we should log the monitor")
    parser.add_argument("--seed", type=int, default=-1)

    # options for hivemind only
    parser.add_argument("--run_hivemind", action="store_true")
    parser.add_argument("--initial_peers", type=str, default="")
    # terminate on epoch OR max_runtime_s or max_hivemind_epochs
    parser.add_argument("--max_hivemind_epochs", type=int, default=-1, required="--run_hivemind" in sys.argv)
    parser.add_argument("--use_local_updates", type=ast.literal_eval, default=False, required="--run_hivemind" in sys.argv)
    parser.add_argument("--target_batch_size", type=int, required="--run_hivemind" in sys.argv)
    parser.add_argument("--matchmaking_time", type=float, required="--run_hivemind" in sys.argv)
    parser.add_argument("--averaging_timeout", type=float, required="--run_hivemind" in sys.argv)
    parser.add_argument("--average_state_every", type=int, required="--run_hivemind" in sys.argv)
    # queen only parameters
    parser.add_argument("--announce_maddr_ip", type=str, default=None, required=False)
    parser.add_argument("--announce_maddr_port", type=str, default=None, required=False)
    parser.add_argument("--is_first_peer", default=False,  type=ast.literal_eval, required=False)
    parser.add_argument("--pre_start_wait_time_s", type=int, default=0, required=False)
    # hivemind performance specific 
    parser.add_argument("--grad_compression", type=str, default="hivemind.NoCompression", required=False)
    parser.add_argument("--state_averaging_compression", type=str, default="hivemind.NoCompression", required=False)
    parser.add_argument("--delay_state_averaging", default=False, type=ast.literal_eval, required=False)
    parser.add_argument("--delay_optimizer_step", default=False, type=ast.literal_eval, required=False)

    # fmt: on

    # don't throw if we are parsing unknown args
    args, unknown = parser.parse_known_args()

    config = {
        "type": "bee",
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "",
        **vars(args),
    }
    config.update(
        {"optimizer_params": json.loads(utils.jsonify(config.get("optimizer_params")))}
    )
    config.update(
        {"scheduler_params": json.loads(utils.jsonify(config.get("scheduler_params")))}
    )
    logger.info(config)

    name = f"bee-{args.host}-{args.run_name}"
    wandb.init(
        config=config,
        name=name,
    )
    start_monitor(config)
    start_bee(config)
