import importlib
import torch
import hivemind


def select_grad_compression(compression_name: str = None) -> hivemind.CompressionBase:
    if compression_name == None or compression_name == "":
        return hivemind.NoCompression()

    opt = setup_grad_compression(compression_name)

    if opt == None:
        raise ValueError(f'Selected compression "{compression_name}" was not found')

    return opt


def setup_grad_compression(compression_name: str) -> hivemind.CompressionBase:
    # e.g. compression_name = 'torch.optim.SGD'
    if "." in compression_name:
        mod_name, func_name = compression_name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        compression_function = getattr(mod, func_name)
        opt = compression_function()
    else:
        # or compression_name = 'Customcompression'
        if compression_name in locals():
            opt = locals()[compression_name]()
    return opt
