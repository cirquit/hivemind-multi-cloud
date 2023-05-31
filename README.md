# Artifacts and Reproducibility
## How Can We Train Deep Learning Models Across Clouds and Continents? An Experimental Study

Currently in submission. 

* [**arXiv Link**](TODO)

### Artifacts

Each experiment is named either a **`baseline-X`** or **`hivemind-X`** run, whether it was running on a single node, or with multiple nodes. 
The Hivemind experiments have multiple runs, with the **`trainmonitor`** tracking the training progress, and the **`bee`**-processes, that run each on a single node, contribute to the training progress.

All experiments were logged via [**Weights & Biases (W&B)**](wandb.ai) and the experiment names are described in a Google Docs page. The different types of experiments (`Model Suitability`, `Geo-Distributed Performance`, `Multi-Cloud Performance`) are color-coded and correspond to the sections in the paper.

* [**W&B Project**](TODO)
* [**Google Docs Experiment Page**](TODO)

All logs from W&B are downloaded and stored at the [**artifacts/wandb**](artifacts/wandb) subdirectory.
The paper figures are generated via the Jupyter Notebook in [**notebooks**](notebooks) subdirectory.
If the artifacts subdirectory is deleted, running the Jupyter Notebooks will download all logs on-demand from W&B again.
When the paper figures are regenerated, the paper can be compiled with `make` in the [**paper**](paper) subdirectory.

The only exception are the `iperf` and `ping` logs, which were gathered manually with [**code/network-profile.sh**](code/network-profile.sh) and can be found at [**artifacts/networking**](artifacts/networking) with their respective experimental setup name (check the Google Docs).

### Reproducibility

All experiments can be reproduced automatically via ansible if the VMs are available.

To fully reproduce all experiments, you need an account at these services, a valid payment method and an SSH key.
It took us multiple calls with representatives from the hyperscale clouds to get the permission to spawn eight VMs in different zones, so be prepared for it to take a while.

* [**backblaze.com**](backblaze.com) - Dataset storage provider (any S3 storage will suffice).
* [**lambdalabs.com**](lambdalabs.com) - Main cloud provider for the model suitability experiments. Also used for hybrid-cloud experiments.
* [**cloud.google.com**](cloud.google.com) - Main cloud provider for geo-distributed experiments, needs access to at least 8 VMs in the US, EU, ASIA and OCE regions. Also used for hybrid-cloud experiments.
* [**aws.amazon.com**](aws.amazon.com) - Additional cloud provider for multi-cloud experiments, needs access to at least 2 VMs in the US.
* [**portal.azure.com**](portal.azure.com) - Additional cloud provider for multi-cloud experiments, needs access to at least 2 VMs in the US.
* On-premise hardware in the EU, preferrably an RTX8000 and a DGX-2, but any non-hyperscale-cloud hardware will do, to showcase different throughputs. Based on the computational capability and bandwidth, results may differ.

---

#### 1. Prepare The Datasets

To start the download and preprocessing, you need a Kaggle account and the credentials as JSON.
Move your downloaded `kaggle.json` to `hivemind-multi-cloud/bin/kaggle.json`

We upload the shards to B2, but you can pick whatever storage you like as long as [webdataset](https://github.com/webdataset/webdataset) can handle it (S3, wget, etc.).

Modify the environment variables in [**bin/00-environment-variables.sh**](bin/00-environment-variables.sh) to point to
  * `IMAGENET_DATASET_PATH` - (default) `/tmp/imagenet-dataset`
  * `WIKIPEDIA_DATASET_PATH` - (default) `/tmp/wikipedia-dataset`
  * `IMAGENET_B2_BUCKETNAME` - (default) `imagenet` 
  * `WIKIPEDIA_B2_BUCKETNAME` - (default) `wikipedia`

```bash
> cd bin
> source 00-environment-variables.sh
> ./download-datasets.sh
> ./shard-wikipedia-dataset.sh 
> ./shard-imagenet-dataset.sh
```

Create a bucket in B2 with the names from `IMAGENET_B2_BUCKETNAME` and `WIKIPEDIA_B2_BUCKETNAME`. If you use other names, adapt the upload scripts accordingly.

```bash
> echo "MY_B2_ACCOUNT_INFO" > ~/.b2_account_info
> ./upload-wikipedia-dataset.sh
> ./upload-imagenet-dataset.sh
```

After uploading the datasets, get the respective download links by going to "Browse Files" -> "Buckets" -> "imagenet/wikipedia" -> "Click on a shard" -> "Friendly URL".
Exchange all the shards URLs in the dataloading code for the CV and NLP experiments in [**code/datasets.py**](code/datasets.py).

---

#### 2. Spawn The VMs

Modify the environment variables in [**bin/00-environment-variables.sh**](bin/00-environment-variables.sh) to point to
  * `LAMBDALABS_SECRET_KEY` - 

While we do have scripts to automatically spawn GC/AWS/Azure VMs, they are highly specific to the different vendors project names, how they handle SSH keys and networks. We recommend to follow create the following setup and copy the resulting command by yourself.

Configure the network to enable ICMP and Ingress TCP 45555. Enable Spot Pricing. Add at least 30GB of storage space.

Template names:
- GC: `n1-standard-8`, User: `ubuntu`
- AWS: `gd4n.2xlarge`, User: `ubuntu`
- Azure: `NC4as_T4_v3` User: `ubuntu`, Offer: `nvidia-gpu-optimized-vmi-a10`, Plan: `nvidia_base_a10_vmi_22_08_gen2`
- LambdaLabs: Full A10 node, User: `ubuntu`



1. Install ansible - [**Official Docs**](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html)
2. Install task - [**Official Docs**](https://taskfile.dev/installation/)
3. Install clusterssh - `sudo apt install clusterssh`
4. Spawn the VMs
- 4.1 LambdaLabs - A10s - `cd bin && ./start-lambda-resources.sh` (single GPU)
- 4.2 Google Cloud - T4 - `cd bin && ./start-gc-resources.sh` (check out the regions that are pre-selected)
- 4.3 Azure / AWS - T4 - Start manually as we only use them twice.
5. Get the respective IPs and update your `~/.ssh/config` and name the VMs in a nice way, e.g., `gc-t4-1-us`
6. Log into them via `clusterssh`
7. Install CUDA on GC and AWS (Azure has their own template, LambdaLabs works out of the box). Both GC and AWS did not like the official nvidia-docker, so this is where we're at.
- 7.1 GC
```bash
sudo apt install wget libxml2 build-essential psmisc file rsync tmux git linux-headers-`uname -r` -y
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run --silent
```
- 7.2 AWS
```bash
sudo apt install wget libxml2 build-essential psmisc file rsync tmux git linux-headers-5.15.0-1033-aws iperf -y
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run --silent
```

---

#### 3. Run The Experiments

**Preliminary**:
- You have a Weights&Biases account.
- CUDA installed.
- You can log in via `ssh gc-t4-1-us`.
- Dataset is uploaded to storage, the download links are updated at [**code/datasets.py**](code/datasets.py)
- Ansible and Task programs are installed

1. Modify the [**ansible/inventory**](ansible/inventory) to use as many VMs as you'd like.
2. The `queen` is the initial VM to which all the peers initially connect to; update it's global IP.
3. Update the inventory user variable, if your default user is not `ubuntu`, otherwise follow the default.
4. Prepare a `trainmonitor` VM which scrapes the DHT, it can be any VM with at least 4 cores (running locally also possible, replace the VM name with `localhost`).
5. Update the `code/.env` file with your WANDB_API_KEY and WANDB_PROJECT and WANDB_ENTITY.

**Model Suitability Experiments**
1. Run the baseline experiments (single A10 GPU) `./run-baseline-experiments.sh`
2. Run the 2xA10 experiments (two A10 GPU) `./run-2xA10-experiments.sh`
3. Run the 3,4,8xA10 experiments (3,4,8 A10 GPU respectivley) `./run-multi-A10-experiments.sh`

**Geo-distributed Experiments**
1. T4 Baseline (GC 1xT4) `./run-cv-nlp-32k-single.sh`
2. 2xT4        (GC 2xT4, same zone) `./run-cv-nlp-32k-multi.sh`
3. 3xT4        (GC 3xT4, same zone) `./run-cv-nlp-32k-multi.sh`
4. 4xT4        (GC 4xT4, same zone) `./run-cv-nlp-32k-multi.sh`
5. 6xT4        (GC 6xT4, same zone) `./run-cv-nlp-32k-multi.sh`
6. 8xT4        (GC 8xT4, same zone) `./run-cv-nlp-32k-multi.sh`
7. 1xT4+1xT4   (GC us-west, eu-central) `./run-cv-nlp-32k-multi.sh`
8. 2xT4+2xT4   (GC us-west, eu-central) `./run-cv-nlp-32k-multi.sh`
9. 2xT4+4xT4   (GC us-west, eu-central) `./run-cv-nlp-32k-multi.sh`
10. 4xT4+4xT4   (GC us-west, eu-central) `./run-cv-nlp-32k-multi.sh`
11. 1xT4+1xT4+1xT4   (GC us-west, eu-central, asia-east) `./run-cv-nlp-32k-multi.sh`
12. 2xT4+2xT4+2xT4   (GC us-west, eu-central, asia-east) `./run-cv-nlp-32k-multi.sh`
13. 1xT4+1xT4+1xT4+1xT4 (GC us-west, eu-central, asia-east, australia-southeast) `./run-cv-nlp-32k-multi.sh`
14. 2xT4+2xT4+2xT4+2xT4 (GC us-west, eu-central, asia-east, australia-southeast) `./run-cv-nlp-32k-multi.sh`

**Multi-cloud Experiments**
1. 2xT4+2xT4 (GC us-west, AWS us-west) `./run-cv-nlp-32k-multi.sh`
2. 2xT4+2xT4 (GC us-west, Azure us-west) `./run-cv-nlp-32k-multi.sh`

**Hybrid-cloud Experiments**

1. RTX8000 `./run-cv-nlp-32k-single.sh`
2. RTX8000 + 1xT4 (GC EU) `./run-cv-nlp-32k-multi.sh`
3. RTX8000 + 2xT4 (GC EU) `./run-cv-nlp-32k-multi.sh`
4. RTX8000 + 4xT4 (GC EU) `./run-cv-nlp-32k-multi.sh`
5. RTX8000 + 8xT4 (GC EU) `./run-cv-nlp-32k-multi.sh`
6. RTX8000 + 1xT4 (GC US) `./run-cv-nlp-32k-multi.sh`
7. RTX8000 + 2xT4 (GC US) `./run-cv-nlp-32k-multi.sh`
8. RTX8000 + 4xT4 (GC US) `./run-cv-nlp-32k-multi.sh`
9. RTX8000 + 8xT4 (GC US) `./run-cv-nlp-32k-multi.sh`
10. RTX8000 + 1xA10 (Lambda US) `./run-cv-nlp-32k-multi.sh`
11. RTX8000 + 2xA10 (Lambda US) `./run-cv-nlp-32k-multi.sh`
12. RTX8000 + 4xA10 (Lambda US) `./run-cv-nlp-32k-multi.sh`
13. RTX8000 + 8xA10 (Lambda US) `./run-cv-nlp-32k-multi.sh`
14. DGX-2 `./run-cv-nlp-32k-single.sh`
15. DGX-2 + 1xT4 (GC EU) `./run-cv-nlp-32k-multi.sh`
16. DGX-2 + 2xT4 (GC EU) `./run-cv-nlp-32k-multi.sh`
17. DGX-2 + 4xT4 (GC EU) `./run-cv-nlp-32k-multi.sh`
18. DGX-2 + 8xT4 (GC EU) `./run-cv-nlp-32k-multi.sh`
19. DGX-2 + 1xT4 (GC US) `./run-cv-nlp-32k-multi.sh`
20. DGX-2 + 2xT4 (GC US) `./run-cv-nlp-32k-multi.sh`
21. DGX-2 + 4xT4 (GC US) `./run-cv-nlp-32k-multi.sh`
22. DGX-2 + 8xT4 (GC US) `./run-cv-nlp-32k-multi.sh`
23. DGX-2 + 1xA10 (Lambda US) `./run-cv-nlp-32k-multi.sh`
24. DGX-2 + 2xA10 (Lambda US) `./run-cv-nlp-32k-multi.sh`
25. DGX-2 + 4xA10 (Lambda US) `./run-cv-nlp-32k-multi.sh`
26. DGX-2 + 8xA10 (Lambda US) `./run-cv-nlp-32k-multi.sh`

I highly recommend noting down the specific names of the experiments in the Google Docs to be able to regenerate the figures. 

---

### 4. Recreate the figures

The notebooks in [**bin/notebooks**](bin/notebooks) will regenerate by default the paper data.
To fully reproduce the results, you will need to replace my experiment names with the newly generated ones.
After that, simply running them in a local virtualenv should create the figures in-place in the [**paper/figures**](paper/figures) directory.

A single `make` in the `paper` subdirectory should recreate all figures.

---

Please contact at `alex.isenko@tum.de` for any questions regarding reproducibility.