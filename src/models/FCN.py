import torch
import torch.nn.functional as F
from torch import nn


class FCNFeatureExtractor(nn.Module):
    def __init__(self, n_in_channels: int, padding_mode: str = "replicate", seq_len=None):
        super().__init__()
        self.n_in_channels = n_in_channels
        kernel_size = [8, 5, 3]
        stride = 1
        paddings = []
        for k in kernel_size:
            paddings.append(max(int(((seq_len - 1) * stride + k - seq_len) / 2), 0))
        self.instance_encoder = nn.Sequential(
            ConvBlock(n_in_channels, 128, 8, padding_mode=padding_mode),
            ConvBlock(128, 256, 5, padding_mode=padding_mode),
            # ConvBlock(256, 256, 5, padding_mode=padding_mode),
            ConvBlock(256, 128, 3, padding_mode=padding_mode)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = 5
        # if x.shape[-1] >= min_len:
        if len(x.shape) == 4:
            x.squeeze(1)
        return self.instance_encoder(x)
        # else:
        #     padded_x = manual_pad(x, min_len)
        #     return self.instance_encoder(padded_x)


class ConvBlock(nn.Module):
    """Convolutional module: Conv1D + BatchNorm + (optional) ReLU."""

    def __init__(
            self,
            n_in_channels: int,
            n_out_channels: int,
            kernel_size: int,
            padding_mode: str = "replicate",
            include_relu: bool = True,
            stride: int = 1,
            padding: int = 0
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
                stride=stride
            ),
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        if include_relu:
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return out


def manual_pad(x: torch.Tensor, min_length: int) -> torch.Tensor:
    """
    Manual padding function that pads x to a minimum length with replicate padding.
    PyTorch padding complains if x is too short relative to the desired pad size, hence this function.

    :param x: Input tensor to be padded.
    :param min_length: Length to which the tensor will be padded.
    :return: Padded tensor of length min_length.
    """
    # Calculate amount of padding required
    pad_amount = min_length - x.shape[-1]
    # Split either side
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    # Pad left (replicate first value)
    pad_x = F.pad(x, [pad_left, 0], mode="constant", value=x[:, :, 0].item())
    # Pad right (replicate last value)
    pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=x[:, :, -1].item())
    return pad_x


class FCNDecoder(nn.Module):
    def __init__(self, n_out_channels: int, padding_mode: str = "zeros", args=None):
        super().__init__()
        kernel_size = [8, 5, 3]
        stride = 1
        paddings = []
        seq_len = args.sample_len
        self.args = args
        # self.patch_stride = int(args.patch_len*args.patch_stride)
        self.patch_stride = int(self.args.shapelet_stride)

        n_out_channels = args.patch_len * args.nvars
        seq_len = int((args.sample_len - args.patch_len) / int(self.patch_stride)) + 1
        for k in kernel_size:
            paddings.append(max(int(((seq_len - 1) * stride + k - seq_len) / 2), 0))

        changed_length = seq_len + 13 - 2 * sum(paddings)
        changed_length *= n_out_channels

        self.instance_decoder = nn.Sequential(
            DeconvBlock(128, 256, 3, padding_mode=padding_mode, padding=paddings[-1]),
            DeconvBlock(256, 128, 5, padding_mode=padding_mode, padding=paddings[-2]),
            DeconvBlock(128, n_out_channels, 8, padding_mode=padding_mode, padding=paddings[-3],
                        include_relu=True),
            # Optionally add an activation function at the end if needed
        )
        self.linear = nn.Linear(changed_length, args.sample_len)
        # self.linear = nn.Linear(changed_length, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.args.patch_type == 'before_encode' and self.args.patch:
            return self.linear(self.instance_decoder(x).view(x.shape[0], 1, -1))
       return self.linear(self.instance_decoder(x))


class DeconvBlock(nn.Module):
    """Deconvolutional module: ConvTranspose1D + BatchNorm + (optional) ReLU."""

    def __init__(
            self,
            n_in_channels: int,
            n_out_channels: int,
            kernel_size: int,
            padding_mode: str = "zeros",
            include_relu: bool = True,
            stride: int = 1,
            padding: int = 0
    ) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                stride=stride
            ),
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        if include_relu:
            layers.append(nn.ReLU())
        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.deconv_block(x)
        return out
