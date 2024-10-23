import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Note: Implementation follows the U-Net architecture as described in the original paper:
# https://arxiv.org/abs/1505.04597
# Possible improvement: use bilinear upsampling instead of transposed convolution.


class DoubleConv(nn.Module):
    """
    A module that performs two consecutive convolution operations, each followed by batch normalization and ReLU activation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.


    Returns:
        torch.Tensor: Output tensor after applying the double convolution.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x) -> torch.Tensor:
        return self.double_conv(x)


class Encoder(nn.Module):
    """
    A module that performs the encoding part of the U-Net architecture.

    Args:
        in_ch (int): Number of input channels.
        num_filters (list): List of number of filters in each layer.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.

    Returns:
        torch.Tensor: Output tensor after encoding.
        list: List of skip connections.
    """

    def __init__(self, in_ch, num_filters, kernel_size=3, stride=1, padding=1):
        super(Encoder, self).__init__()
        layers = []
        for num_filter in num_filters:
            layers.append(DoubleConv(in_ch, num_filter, kernel_size, stride, padding))
            in_ch = num_filter
        self.encoder = nn.ModuleList(layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x, skip_connections


class Decoder(nn.Module):
    """
    A module that performs the decoding part of the U-Net architecture.

    Args:
        num_filters (list): List of number of filters in each layer.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.

    Returns:
        torch.Tensor: Output tensor after decoding.
    """

    def __init__(self, num_filters, kernel_size=3, stride=1, padding=1):
        super(Decoder, self).__init__()
        layers = []
        for num_filter in reversed(num_filters):
            layers.append(
                nn.ConvTranspose2d(num_filter * 2, num_filter, kernel_size=2, stride=2)
            )
            layers.append(
                DoubleConv(num_filter * 2, num_filter, kernel_size, stride, padding)
            )
        self.decoder = nn.ModuleList(layers)

    def forward(self, x, skip_connections):
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections.pop()
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder[i + 1](x)
        return x


class Bottleneck(nn.Module):
    """
    A module that performs the bottleneck part of the U-Net architecture.

    Args:
        num_filters (list): List of number of filters in each layer.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.

    Returns:
        torch.Tensor: Output tensor after bottleneck.
    """

    def __init__(self, num_filters, kernel_size=3, stride=1, padding=1):
        super(Bottleneck, self).__init__()
        self.bottleneck = DoubleConv(
            num_filters[-1], num_filters[-1] * 2, kernel_size, stride, padding
        )

    def forward(self, x):
        return self.bottleneck(x)


class UNet(nn.Module):
    """
    A module that implements the U-Net architecture.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        num_filters (list): List of number of filters in each layer.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.

    Returns:
        torch.Tensor: Output tensor after passing through the U-Net.
    """

    def __init__(
        self,
        in_ch=2,
        out_ch=4,
        num_filters=[64, 128, 256, 512],
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_ch, num_filters, kernel_size, stride, padding)
        self.bottleneck = Bottleneck(num_filters, kernel_size, stride, padding)
        self.decoder = Decoder(num_filters, kernel_size, stride, padding)
        self.final_conv = nn.Conv2d(num_filters[0], out_ch, kernel_size=1)

        # Additional activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # Additional conv layer to adjust output dimensions
        self.adjust_dims = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = self.final_conv(x)
        x = self.adjust_dims(x)

        # Now x has shape [batch_size, out_ch, H', W']
        # Split the outputs
        swh_mwp_outputs = x[:, :2, :, :]  # First two channels for SWH and MWP
        mwd_outputs = x[:, 2:, :, :]  # Last two channels for MWD sine and cosine

        # Apply activation functions
        swh_mwp_outputs = self.sigmoid(swh_mwp_outputs)  # Outputs in [0, 1]
        mwd_outputs = self.tanh(mwd_outputs)  # Outputs in [-1, 1]

        # Concatenate outputs back together
        outputs = torch.cat((swh_mwp_outputs, mwd_outputs), dim=1)
        return outputs


def test():
    # Test with the specific input and output dimensions
    x = torch.randn((1, 2, 25, 33))  # Batch size 1, 2 channels (u and v), 25x33 grid
    model = UNet()
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape == (
        1,
        4,
        13,
        17,
    ), f"Expected shape (1, 4, 13, 17), got {preds.shape}"
    print("UNet test passed!")


if __name__ == "__main__":
    test()
