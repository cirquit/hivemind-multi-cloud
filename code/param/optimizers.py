from functools import partial
import importlib
import torch


def select_optimizer(
    optimizer_name: str = None,
    optimizer_params: dict = {},
    model: torch.nn.Module = None,
    lambda_return: bool = False
) -> torch.optim.Optimizer:
    if optimizer_name == None:
        raise ValueError("Optimizer has not been selected for this run")

    opt = setup_optimizer(optimizer_name, optimizer_params, model, lambda_return)

    if opt == None:
        raise ValueError(f'Selected optimizer "{optimizer_name}" was not found')

    return opt


def setup_optimizer(
    optimizer_name: str,
    optimizer_params: dict,
    model: torch.nn.Module,
    lambda_return: bool,
    ):
    # e.g. optimizer_name = 'torch.optim.SGD'
    if "." in optimizer_name:
        mod_name, func_name = optimizer_name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        optimizer_function = getattr(mod, func_name)
        if model is None: # probably auxiliary peer
            opt = partial(optimizer_function, **optimizer_params)
        else:
            # to enable delayed parameter updates
            if lambda_return:
                opt = lambda params: optimizer_function(params, **optimizer_params)
            else:
                opt = optimizer_function(model.parameters(), **optimizer_params)

            
    else:
        # or optimizer_name = 'CustomOptimizer'
        if optimizer_name in globals():
            if model is None: # probably auxiliary peer
                opt = partial(globals()[optimizer_name], **optimizer_params)
            else:
                opt = globals()[optimizer_name](model.parameters(), **optimizer_params)
        else:
            raise ValueError("Optimizer was not found in globals().")
    return opt
