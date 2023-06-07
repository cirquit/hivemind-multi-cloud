import time

from hivemind.utils.logging import get_logger, use_hivemind_log_handler

import wandb

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def cuda_sync():
    """Waits until cuda is fully finished (for reproducibility purposes)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimeIt(object):
    def __init__(self):
        self.delta_time_s = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        self.delta_time_s = end - self.start

    def delta_time_s(self):
        return self.delta_time_s


class CUDATimeIt(object):
    def __init__(self):
        self.delta_time_s = 0

    def __enter__(self):
        cuda_sync()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        cuda_sync()
        end = time.time()
        self.delta_time_s = end - self.start

    def delta_time_s(self):
        return self.delta_time_s


class WandbTimeIt(object):
    def __init__(
        self, name: str, cuda: bool = False, commit: bool = False, verbose: bool = False
    ):
        """Times the inner code with perf_counter and process_time and logs to wandb
        perf_counter counts actual time, process_time counts the process time,
        i.e. no sleep included
        https://docs.python.org/3/library/time.html#time.process_time
        """
        self.delta_time_perf_s = 0
        self.name_perf = "02_timing/" + name + "_time_s"
        self.commit = commit
        self.cuda = cuda
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            logger.info(f"Entered {self.name_perf}")
        if self.cuda:
            cuda_sync()
        self.start_perf = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.verbose:
            logger.info(f"Exited {self.name_perf}")
        if self.cuda:
            cuda_sync()
        end_perf = time.perf_counter()
        self.delta_time_perf_s = end_perf - self.start_perf
        wandb.log(
            {
                self.name_perf: self.delta_time_perf_s,
            },
            commit=self.commit,
        )

    def delta_time_s(self):
        """The default is perf_counter time"""
        return self.delta_time_perf_s
