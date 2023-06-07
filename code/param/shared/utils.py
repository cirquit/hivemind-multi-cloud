import copy
import json
import os
import threading
import random
import subprocess as sp
import time

from typing import Callable, List, Tuple, Union
from shared.monitor import Monitor
import hivemind
import numpy as np
import torch
import wandb
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def start_monitor(config: dict):
    def inner():
        mon = Monitor()
        wandb.log(mon.get_static_info())
        monitor_log_frequency_ms = config.get("monitor_log_frequency_ms")
        monitor_log_frequency_s = monitor_log_frequency_ms / 1000
        while True:
            wandb.log({**mon.get_sys_info()})
            time.sleep(monitor_log_frequency_s)

    t = threading.Thread(target=inner)
    t.daemon = True
    t.start()


def shutdown_optimizer(opt: Union[torch.optim.Optimizer, hivemind.Optimizer]):
    if isinstance(opt, hivemind.Optimizer):
        opt.shutdown()
        opt.dht.shutdown()
        logger.info("Waiting for DHT to terminate...")
        opt.dht.join(timeout=10)
        logger.info("DHT Terminated successfully")


def parse_compression(value: str):
    """ """
    return getattr(CompressionType, value)


def write_maddrs_to_file(server: hivemind.moe.Server):
    """Writing a file to copy via ansible and distribute to queen"""
    visible_maddrs = [str(maddr) for maddr in server.dht.get_visible_maddrs()]
    wandb.config.update({"visible_maddrs": visible_maddrs})
    write_conf_to_file(wandb.config)


def seed_everything(seed: int):
    """Sets the seed globally for cuda/numpy/random/torch
    Copied from https://clay-atlas.com/us/blog/2021/08/24/pytorch-en-set-seed-reproduce/
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def disable_torch_debugging():
    """Torch runs by default with debugging enabled.
    Disable on performance runs.
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-debugging-apis
    """

    def set_debug_apis(state: bool = False):
        torch.autograd.profiler.profile(enabled=state)
        torch.autograd.profiler.emit_nvtx(enabled=state)
        torch.autograd.set_detect_anomaly(mode=state)

    set_debug_apis(False)


def write_conf_to_file(wandb_conf):
    config = dict(wandb_conf)
    if "optim_cls" in config.keys():
        config.pop("optim_cls")
    with open(".configuration.json", "w") as f:
        f.write(json.dumps(config))


def cuda_sync():
    """Waits until cuda is fully finished (for reproducibility purposes)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def drop_io_cache(page_cache=True, dentries_and_inodes=True):
    """Drops the virtual memory cache so we don't have any accidental cache hits when running the
    same experiment multiple times
    """
    code = 0
    if page_cache:
        code += 1
    if dentries_and_inodes:
        code += 2
    drop_cmd = "echo {} > /drop_caches".format(code)
    cmd_array = ["sudo", "sh", "-c", drop_cmd]
    try:
        sp.run(cmd_array, check=True)
    except sp.CalledProcessError as e:
        print("Error: Could not drop caches!")
        print(f"PythonError: {e}")


def test_model(model, dataloader, device):
    """Returns the accuracy of the model"""
    correct = 0
    total = 0
    model.train(mode=False)
    model.to(device)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        #            logger.info(f"Prediction: {prediction}")
        #            logger.info(f"Labels: {labels}")
        logger.info(f"Validation - Correct: {correct}, Total: {total}")
    model.train(mode=True)
    return 100 * correct / total


def jsonify(s):
    """Addes double quotes to a dictionary without quotes around parameters due to ansible
    swallowing any kind of escaped quotes"""
    result = []
    opened = False
    for c in s:
        # inside a name
        if (c.isalpha() or c == "_") and opened:
            pass
        # not inside a name
        elif (c.isalpha() or c == "_") and not opened:
            opened = True
            result.append('"')
        # name finished
        elif opened:
            opened = False
            result.append('"')
        result.append(c)
    return "".join(result)
