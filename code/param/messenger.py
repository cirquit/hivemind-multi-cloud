import argparse
import os
import time

import hivemind
# import wandb
from dotenv import load_dotenv
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from shared.utils import write_conf_to_file

from shared.monitor import Monitor

load_dotenv(override=True)

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

# the messenger handles the initial DHT server


def start_messenger(config: dict):
    monitor = Monitor()

    # die after 1200 seconds since we can't find anything
    MAX_N_RETIRES = 240
    REFRESH_TIME_S = 5
    PORT = 45555

    dht = hivemind.DHT(
        initial_peers=config.get("initial_peers"),
        # listen globally
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{PORT}", f"/ip4/0.0.0.0/udp/{PORT}/quic"],
        start=True,
        # disable P2P timeout or it will probably die while the bees try to connect to it
        idle_timeout=MAX_N_RETIRES * REFRESH_TIME_S,
        use_relay=True,
        use_auto_relay=True,
        client_mode=False,
    )

    visible_maddrs = [str(maddr) for maddr in dht.get_visible_maddrs()]

    for maddr in visible_maddrs:
        logger.info(maddr)

    #    wandb.config.update({"visible_maddrs": visible_maddrs})
    config["visible_maddrs"] = visible_maddrs
    write_conf_to_file(config)

    try:
        current_retries = 0
        while dht.is_alive():
            # progress_dict = dht.get(
            #    wandb.config.get("run_name") + "_progress", latest=True
            # )
            # if progress_dict is None:
            #    current_retries += 1
            #    if current_retries >= MAX_N_RETIRES:
            #        break
            # else:
            #    current_retries = 0
            # wandb.log(monitor.get_sys_info())
            time.sleep(REFRESH_TIME_S)
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt or SIGINT, shutting down")
    finally:
        dht.shutdown()
        logger.info("Waiting for DHT to terminate...")
        dht.join(timeout=10)
        logger.info("DHT Terminated successfully")
        # wandb.log(monitor.get_sys_info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start up a messenger instance")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host where the machine is currently running on",
    )
    parser.add_argument("--initial_peers", default="", help="Initial DHT peers")
    parser.add_argument(
        "--run_name",
        help="Name of current run",
        required="WANDB_DISABLED" not in os.environ,
    )

    # don't throw if we are parsing unknown args
    args, unknown = parser.parse_known_args()

    name = f"messenger-{args.host}-{args.run_name}"
    config = {
        "type": "messenger",
        **vars(args),
    }
    #    wandb.init(config=config, name=name)

    start_messenger(config)
