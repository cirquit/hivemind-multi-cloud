import os
import time
from typing import Callable, Tuple

import torch
from transformers import AutoTokenizer
import webdataset as wds
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from torchvision import datasets, transforms
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

import wandb

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

noop = lambda *_, **__: None

def select_dataloader(
    dataset_name: str,
    dataset,
    batch_size_per_step: int,
    num_workers: int
):    
    # best practice from https://webdataset.github.io/webdataset/sharding/
    if isinstance(dataset, wds.WebDataset):
        # if running LLMs, the dataloader is configured slightly differently due to collate
        if "wiki_roberta_wds_online" in dataset_name:
            
            if dataset_name == "wiki_roberta_wds_online_base":
                tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            elif dataset_name == "wiki_roberta_wds_online_large":
                tokenizer = AutoTokenizer.from_pretrained('roberta-large')
            elif dataset_name == "wiki_roberta_wds_online_xlm":
                tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=data_collator,
                batch_size=batch_size_per_step,
                num_workers=num_workers,
            )
        # non LLMs WDS config
        else:
            train_dataloader = torch.utils.data.DataLoader(
                dataset.batched(batch_size_per_step),
                batch_size=None,
                num_workers=num_workers,
            )
    # not WDS
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size_per_step,
            num_workers=num_workers,
        )
    return train_dataloader


def select_dataset(
    dataset_name: str = None, data_folder: str = "", dont_load: bool = False, **kwargs
):
    if dataset_name == None:
        raise ValueError("Dataset has not been selected for this run")

    if data_folder == "" and not dont_load:
        raise ValueError("Cannot use an empty data folder path")

    if dataset_name.lower() == "imagenet":
        if dont_load:
            return None, None, 1000, 3, noop
        return setup_imagenet(data_folder=data_folder, **kwargs)
    if dataset_name.lower() == "imagenet_wds":
        if dont_load:
            return None, None, 1000, 3, noop
        return setup_imagenet_wds(data_folder=data_folder, **kwargs)
    if dataset_name.lower() == "imagenet_wds_online":
        if dont_load:
            return None, None, 1000, 3, noop
        return setup_imagenet_wds_online(**kwargs)
    if dataset_name.lower() == "mnist":
        if dont_load:
            return None, None, 10, 1, noop
        return setup_mnist(data_folder=data_folder)

    if "wiki_roberta_wds_online" in dataset_name.lower():
        if dont_load:
            return None, None, None, 1, noop
        return setup_wikipedia_wds_roberta_online(dataset_name = dataset_name.lower())

    raise ValueError(f'Selected dataset "{dataset_name}" not found')

def setup_wikipedia_wds_roberta_online(dataset_name):
    """ """
    cache_dir = "/tmp/"
    shard_ids = ["ENTER THE SHARDED WIKI URLS HERE",
                 "ENTER THE SHARDED WIKI URLS HERE",
                 "ENTER THE SHARDED WIKI URLS HERE",]
    shard_urls = map(lambda url: f"pipe:curl -L -s {url} || true", shard_ids)
    dataset = wds.WebDataset(
            urls=shard_urls,
            shardshuffle=False,
            handler=wds.handlers.warn_and_continue)
           # cache_dir=cache_dir)

    if dataset_name == "wiki_roberta_wds_online_base":
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif dataset_name == "wiki_roberta_wds_online_large":
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    elif dataset_name == "wiki_roberta_wds_online_xlm":
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    
    def wikipedia_preprocessing(sample):
        sample = sample["text"].decode()[:5000]
        sample = tokenizer(text=sample,
                           max_length=512,
                           padding="max_length",
                           return_special_tokens_mask=True,
                           truncation=True,
                           return_tensors="pt")
        return sample
    # for some reason some samples were constructed with broken text keys
    dataset = dataset.select(lambda sample: "text" in sample.keys())
    dataset = dataset.map(wikipedia_preprocessing)
    
    return dataset, None, None, None, noop


def setup_imagenet_wds_online(
    **kwargs,
) -> Tuple[wds.WebDataset, wds.WebDataset, int, int, Callable]:
    """ """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    cache_dir = "/tmp/"
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    identity = lambda x: x

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_ids = ["ENTER THE SHARDED IMAGENET URLS HERE",
                 "ENTER THE SHARDED IMAGENET URLS HERE",
                 "ENTER THE SHARDED IMAGENET URLS HERE",]

    train_urls = map(lambda url: f"pipe:curl -L -s {url} || true", train_ids)
    val_urls = map(lambda url: f"pipe:curl -L -s {url} || true", train_ids)

    trainset = (
        wds.WebDataset(
            urls=train_urls,
            shardshuffle=False,
            handler=wds.handlers.warn_and_continue)
            #cache_dir=cache_dir)
    )
    trainset = trainset.select(lambda sample: "jpg" in sample.keys())
    trainset = trainset.decode("pil")
    trainset = trainset.to_tuple("jpg", "cls")
    trainset = trainset.map_tuple(transform_train, identity)       

    valset = (
        wds.WebDataset(
            urls=val_urls,
            shardshuffle=False,
            handler=wds.handlers.warn_and_continue)
            #cache_dir=cache_dir)
    )
    valset = valset.select(lambda sample: "jpg" in sample.keys())
    valset = valset.decode("pil")
    valset = valset.to_tuple("jpg", "cls")
    valset = valset.map_tuple(transform_train, identity)   


    def accuracy_fn(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1 / batch_size))
            return res

    return trainset, valset, 1000, 3, accuracy_fn


def setup_imagenet_wds(
    data_folder: str, **kwargs
) -> Tuple[wds.WebDataset, wds.WebDataset, int, int, Callable]:
    """ """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    identity = lambda x: x

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    trainset = (
        wds.WebDataset(
            urls=data_folder + "/train/imagenet-train-{000000..000050}.tar",
            shardshuffle=50,
        )  # , cache_dir=cache_dir)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform_train, identity)
    )

    valset = (
        wds.WebDataset(
            urls=data_folder + "/val/imagenet-val-{000000..000006}.tar"
        )  # , cache_dir=cache_dir)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform_val, identity)
    )

    def accuracy_fn(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1 / batch_size))
            return res

    return trainset, valset, 1000, 3, accuracy_fn


def setup_imagenet(
    data_folder: str = "",
    **kwargs,
) -> Tuple[datasets.DatasetFolder, datasets.DatasetFolder, int, int, Callable]:
    """
    :returns:
        - traindataset (torchvision.datasets.X)
        - valdataset (torchvision.datasets.X)
        - input_channels
        - accuracy_fn(output, target)
    """

    def accuracy_fn(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1 / batch_size))
            return res

    trainset, valset = setup_imagenet_pytorch(data_folder)

    return trainset, valset, 1000, 3, accuracy_fn


def setup_imagenet_pytorch(data_folder: str = ""):
    """ """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    start_load = time.perf_counter()
    trainset = datasets.ImageFolder(os.path.join(data_folder, "train"), transform_train)
    end = time.perf_counter()

    start = time.perf_counter()
    valset = datasets.ImageFolder(os.path.join(data_folder, "val_prep"), transform_val)
    end_load = time.perf_counter()

    wandb.log(
        {
            "02_timing/train_dataset_load_s": end - start_load,
            "02_timing/val_dataset_load_s": end_load - start,
        }
    )
    return trainset, valset


# Mind the root path: https://stackoverflow.com/a/66936293/12482799
# has to be: <data_folder>/raw/train-images-idx3-ubyte (uncompressed) for each one of:
# ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
# ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
# ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
# ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
def setup_mnist(
    data_folder: str = "",
) -> Tuple[datasets.DatasetFolder, datasets.DatasetFolder, int, int, Callable]:
    """
    :returns:
        - traindataset (torchvision.datasets.X)
        - valdataset (torchvision.datasets.X)
        - input_channels
        - accuracy_fn(output, target)
    """

    def accuracy_fn(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1 / batch_size))
            return res

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    trainset = datasets.MNIST(
        data_folder, train=True, download=True, transform=transform
    )
    valset = datasets.MNIST(
        data_folder, train=False, download=True, transform=transform
    )

    return trainset, valset, len(trainset.classes), 1, accuracy_fn
