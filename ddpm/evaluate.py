
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch
from config import TrainingConfig

def sample_images(config, epoch, pipeline):
    """
    Evaluates the model by generating images from random noise and saves them in a grid format.

    Args:
        config (TrainingConfig): The configuration object containing evaluation settings.
        epoch (int): The current epoch number, used for naming the output file.
        pipeline (DDPMPipeline): The pre-trained pipeline used for generating images.

    Raises:
        ValueError: If the product of rows and cols does not equal config.eval_batch_size.
    """
    # Ensure the product of rows and cols equals the batch size
    rows = 2
    cols = config.eval_batch_size // rows
    if rows * cols != config.eval_batch_size:
        raise ValueError("The product of rows and cols must equal config.eval_batch_size")

    # Sample images from random noise
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed)
    ).images

    # Create a grid of images
    image_grid = make_image_grid(images, rows=rows, cols=cols)

    # Save the image grid
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

if __name__=='__main__':
    pass
    # config = TrainingConfig()
    # pipeline = DDPMPipeline.from_pretrained('model/checkpoint/path')
    # sample_images(config, epoch=1, pipeline=pipeline)    


