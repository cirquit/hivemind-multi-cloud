import importlib

from transformers import RobertaForMaskedLM, RobertaConfig, XLMRobertaForMaskedLM, XLMRobertaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

# Formula to calculate the outputs for a conv layer
# W = Image Width/Height, e.g., 32 for 32x32
# K = Kernel size, e.g., 5 for 5x5
# P = Padding (default 0)
# S = Stride (default 1)
# Outputs = (W-K+2P)/S + 1


def select_model(
    model_name: str = None,
    num_classes=1000,
    input_channels: int = 3
) -> nn.Module:
    """Wrapper around a python global variables selection, raises error if model not found

    :model_name: (str) = None, torchvision.models.resnet18/vgg16 or ConvNetV1
    :input_channels: (int) = 3, image input channels, greyscale = 1, rbg = 3
    :num_classes: (int) = 1000, final layer outputs
    """

    model = None
    if model_name == None:
        raise ValueError("Model has not been selected for this run")

    model = setup_model(
        model_name=model_name, num_classes=num_classes, input_channels=input_channels
    )

    if model == None:
        raise ValueError(f'Selected model "{model_name}" was not found')

    return model


def setup_model(model_name: str, num_classes: int, input_channels: int) -> nn.Module:
    # e.g. model_name = 'torchvision.models.vgg16'
    if "." in model_name:
        mod_name, func_name = model_name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        model_function = getattr(mod, func_name)
        model = model_function(
            pretrained=False, progress=False, num_classes=num_classes
        )
    # or model_name = 'ConvNetV1'
    elif model_name in globals():
        model = globals()[model_name](
            num_classes=num_classes, input_channels=input_channels
        )
    elif model_name == "roberta_mlm_base": # 124,697,433P
        model = RobertaForMaskedLM(
            RobertaConfig(
                vocab_size = 50265,
                max_position_embeddings = 514,
                # hidden_size = 768,
                # num_hidden_layers = 12,
                # num_attention_heads = 12,
                # intermediate_size = 3072
            )
        )
    elif model_name == "roberta_mlm_large": # 355,412,057P
        model = RobertaForMaskedLM(
            RobertaConfig(
                vocab_size = 50265,
                hidden_size = 1024,
                max_position_embeddings = 514,
                intermediate_size = 4096,
                num_attention_heads = 16,
                num_hidden_layers = 24,

            )
        )
    elif model_name == "roberta_mlm_xlm": # 560,142,482P
        model = XLMRobertaForMaskedLM(
            XLMRobertaConfig(
                vocab_size = 250002,
                hidden_size = 1024,
                max_position_embeddings = 514,
                intermediate_size = 4096,
                num_attention_heads = 16,
                num_hidden_layers = 24,
                type_vocab_size = 1

            )
        )
    else:
        return None
    return model


# https://nextjournal.com/gkoehler/pytorch-mnist
class ConvNetV1(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        """
        :input_channels: (int) - greyscale = 1, RBG = 3
        :num_classes: (int) - final layer outputs
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


class ConvNetCIFARV1(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        """
        :input_channels: (int) - greyscale = 1, RBG = 3
        :num_classes: (int) - final layer outputs
        """
        super().__init__()
        # self.activation = activation
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 800)  # x -> 4x
        self.fc3 = nn.Linear(800, 200)  # 4x -> x
        self.fc4 = nn.Linear(200, 80)
        self.fc5 = nn.Linear(80, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class ConvNetMNISTV1(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        """
        :input_channels: (int) - greyscale = 1, RBG = 3
        :num_classes: (int) - final layer outputs
        """
        super().__init__()
        # self.activation = activation
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 200)
        self.fc2 = nn.Linear(200, 80)
        self.fc3 = nn.Linear(80, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
