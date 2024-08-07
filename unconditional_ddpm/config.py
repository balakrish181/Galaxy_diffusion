
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Configuration for training a diffusion model.

    Attributes:
        image_size (int): The resolution of the generated images.
        train_batch_size (int): The batch size for training.
        eval_batch_size (int): The number of images to sample during evaluation.
        num_epochs (int): The total number of training epochs.
        gradient_accumulation_steps (int): The number of steps to accumulate gradients before updating model weights.
        learning_rate (float): The initial learning rate for the optimizer.
        lr_warmup_steps (int): The number of warmup steps for the learning rate scheduler.
        save_image_epochs (int): Interval (in epochs) to save generated images.
        save_model_epochs (int): Interval (in epochs) to save model checkpoints.
        mixed_precision (str): Precision type, 'no' for float32 and 'fp16' for automatic mixed precision.
        output_dir (str): Directory for saving model checkpoints and outputs.
        push_to_hub (bool): Whether to upload the saved model to the Hugging Face Hub.
        hub_model_id (str): The repository name on the Hugging Face Hub.
        hub_private_repo (bool): Whether the repository on the Hugging Face Hub is private.
        overwrite_output_dir (bool): Whether to overwrite the output directory during re-training.
        seed (int): Random seed for reproducibility.
    """
    image_size: int = 256
    train_batch_size: int = 4
    eval_batch_size: int = 8
    num_epochs: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1500
    save_image_epochs: int = 10
    save_model_epochs: int = 30
    mixed_precision: str = "fp16"
    output_dir: str = "ddpm-cosmos-256-2nd_train"
    push_to_hub: bool = False
    hub_model_id: str = "balakrish181/ddpm-cosmos-256-2nd_train"
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0


if __name__=='__main__':
    config = TrainingConfig()
    