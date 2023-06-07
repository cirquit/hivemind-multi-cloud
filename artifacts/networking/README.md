# Network Profiling

**Important: Configure ICMP and TCP 45555 to be open.**

There are two scripts, one for multi-threaded profiling and one for single-threaded profiling, due to the output being formatted differently. Check the global variables in the scripts to modify repetitions, TCP port and the target nodes.

Most importantly, update the `targets` with the target hostname (name of your choosing) and the public IP.

Target system(s): `iperf -s -p 45555`

Host system: `./network-stats.sh <host-system-name>`

After `ping` and `iperf` profiled the latency and bandwidth, it creates a csv file `iperf-profile-<host-system-name>.csv` with the following columns:

```csv
timestamp,local_hostname,target_hostname,target_ip,bandwidth,bandwidth_metric,avg_ping_ms
```

All [logs](/logs) in are named after the experimental set number and their respective run counter from the [Google Docs document](https://docs.google.com/spreadsheets/d/18ZR-8blrsJFppky4uEqXNRqScZbp955yRHkyCnXpxKU/edit?usp=sharing).

