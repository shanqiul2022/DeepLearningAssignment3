from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

         # Feature extractor: (B, 3, 64, 64) -> (B, 128, 8, 8)
        self.features = nn.Sequential(
            # 64x64 -> 64x64
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 64x64 -> 32x32
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 32x32 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32x32 -> 16x16
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 16x16 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 128, 8, 8) -> (B, 8192)
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()
    
    def _init_weights(self):
        # Kaiming init for convs, xavier for linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        feats = self.features(z)            # (B, 128, 8, 8)
        logits = self.classifier(feats)     # (B, num_classes)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)

class DoubleConv(nn.Module):
    """
    conv -> BN -> ReLU -> conv -> BN -> ReLU
    Keeps spatial size the same when padding=1.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    DoubleConv + 2x2 maxpool for downsampling.
    Returns (features_before_pool, pooled_features).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)        # same H, W
        x_down = self.pool(x)   # H/2, W/2
        return x, x_down        # skip, next_input


class UpBlock(nn.Module):
    """
    ConvTranspose2d upsampling + concatenation with skip + DoubleConv.
    Handles arbitrary (even) input sizes using center-crop if needed.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        in_channels: channels of input to up (bottleneck)
        out_channels: channels after upsample; skip has out_channels too
        """
        super().__init__()
        # up: halve channels, double spatial size
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        # after concat, channels = out_channels (skip) + out_channels (up)
        self.conv = DoubleConv(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)  # upsample

        # handle minor mismatches in spatial dims by center cropping skip
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            skip = skip[
                :,
                :,
                diff_y // 2 : skip.size(2) - diff_y // 2,
                diff_x // 2 : skip.size(3) - diff_x // 2,
            ]

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

         # Encoder: downsample spatial dims, increase channels
        self.down1 = DownBlock(in_channels, 16)   # (B,3,H,W) -> (B,16,H/2,W/2)
        self.down2 = DownBlock(16, 32)            # (B,16,H/2,W/2) -> (B,32,H/4,W/4)

        # Bottleneck
        self.bottleneck = DoubleConv(32, 64)

        # Decoder: upsample and fuse with skip connections
        self.up1 = UpBlock(64, 32)   # (B,64,H/4,W/4) -> (B,32,H/2,W/2)
        self.up2 = UpBlock(32, 16)   # (B,32,H/2,W/2) -> (B,16,H,W)

        # Segmentation head: per-pixel logits for 3 classes
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)

        # Depth head: single-channel depth in [0,1]
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),  # ensures output is in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and depth.

        Args:
            x (torch.FloatTensor): (b, 3, h, w), values in [0,1]

        Returns:
            logits: (b, num_classes, h, w)
            depth:  (b, h, w), values in [0, 1]
        """
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        skip1, x = self.down1(z)     # skip1: (B,16,H,W), x: (B,16,H/2,W/2)
        skip2, x = self.down2(x)     # skip2: (B,32,H/2,W/2), x: (B,32,H/4,W/4)

        # Bottleneck
        x = self.bottleneck(x)       # (B,64,H/4,W/4)

        # Decoder with skip connections
        x = self.up1(x, skip2)       # (B,32,H/2,W/2)
        x = self.up2(x, skip1)       # (B,16,H,W)

        # Heads
        logits = self.seg_head(x)        # (B,num_classes,H,W)
        depth_map = self.depth_head(x)   # (B,1,H,W), already in [0,1]

        depth = depth_map.squeeze(1)     # (B,H,W)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference. This is what the grader's metrics use.

        Args:
            x (torch.FloatTensor): (b, 3, h, w), values in [0, 1]

        Returns:
            pred:  (b, h, w) class labels {0,1,2}
            depth: (b, h, w) depth in [0, 1]
        """
        logits, depth = self(x)
        pred = logits.argmax(dim=1)  # (B,H,W)

        # depth is already in [0,1] thanks to Sigmoid in depth_head
        return pred, depth

MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
