
from config import TrainingConfig
from diffusers import UNet2DModel

config = TrainingConfig()

def unet_model(config):
    """
    Creates a UNet2DModel for image generation tasks using the configuration provided.

    Args:
        config (TrainingConfig): The configuration object containing model parameters.

    Returns:
        UNet2DModel: The initialized UNet2DModel with the specified architecture.
    """
    model = UNet2DModel(
        sample_size=config.image_size,  # The target image resolution.
        in_channels=3,  # Number of input channels; 3 for RGB images.
        out_channels=3,  # Number of output channels.
        layers_per_block=2,  # Number of ResNet layers per UNet block.
        block_out_channels=(128, 128, 256, 256, 512, 512, 1024),  # Output channels for each UNet block.
        down_block_types=(
            "DownBlock2D",  # A regular ResNet downsampling block.
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # A ResNet downsampling block with spatial self-attention.
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # A regular ResNet upsampling block.
            "AttnUpBlock2D",  # A ResNet upsampling block with spatial self-attention.
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

if __name__=='__main__':
    model = unet_model(config)
