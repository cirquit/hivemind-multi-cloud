# Notebooks

Prepare the environment:

```bash
conda create --name hivemind-multi-cloud-eval python==3.9 -y
conda activate hivemind-multi-cloud-eval
pip install -r requirements.txt
./start-notebooks.sh
```

If you want to download the logs anew, either delete the ones in `../artifacts/wandb` or set the `LOG_PATH` to a new directory (approx. 50MB), e.g.:

```
export WANDB_LOG_PATH="/tmp/my_custom_log_path"; export NETWORK_LOG_PATH="../../artifacts/networking/logs"; export FIGURE_PATH="../../paper/figures"; jupyter notebook
```
