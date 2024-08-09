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
        layers_per_block=1,  # Number of ResNet layers per UNet block.
        block_out_channels=(32, 64, 64),  # Output channels for each UNet block.
        down_block_types=(
            "DownBlock2D",  # A regular ResNet downsampling block.
            "DownBlock2D",  # A regular ResNet downsampling block.
            "DownBlock2D",  # A regular ResNet downsampling block.
        ),
        up_block_types=(
            "UpBlock2D",  # A regular ResNet upsampling block.
            "UpBlock2D",  # A regular ResNet upsampling block.
            "UpBlock2D",  # A regular ResNet upsampling block.
        ),
    )
    return model

if __name__ == '__main__':
    model = unet_model(config)
