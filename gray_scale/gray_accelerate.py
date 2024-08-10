from config import TrainingConfig


config = TrainingConfig()



from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional
import cv2

class CosmosImageData(Dataset):
    """
    A custom dataset class for loading and transforming images from a specified directory.

    This class is designed to load images from a given directory, apply any specified transformations,
    and provide an interface compatible with PyTorch's data loading utilities.

    Attributes:
        path (str): The directory containing image files.
        paths (List[Path]): A sorted list of Path objects pointing to image files.
        transform (Optional[Callable]): A callable transformation to apply to each image.
    """

    def __init__(self, path: str, transform: Optional[Callable] = None):
        """
        Initializes the dataset with the path to the images and an optional transform.

        Args:
            path (str): The directory containing image files.
            transform (Optional[Callable]): A transformation to apply to each image, such as resizing or normalization.
        """
        self.path = path
        self.paths = sorted(Path(self.path).glob('*.png'))
        self.transform = transform
        
    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Image.Image:
        """
        Retrieves an image from the dataset and applies the specified transformations.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            Image.Image: The transformed image.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        img = Image.open(self.paths[idx])
        img = img.convert('L')
        
        if self.transform:
            img = self.transform(img)
            
        return img


import torch
import torchvision,torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_data(image_path:Path):

    img_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor()
    ])

    data = CosmosImageData(image_path,transform=img_transform)
    data_loader = DataLoader(data,batch_size=config.train_batch_size,shuffle=True)

    return data_loader

from model import unet_model
from diffusers.optimization import get_cosine_schedule_with_warmup

def load_train_objs(train_dataloader:DataLoader):

    model = unet_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    return model,optimizer,lr_scheduler

from diffusers import DDPMScheduler

def scheduler():
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    

    return noise_scheduler


from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


from accelerate import Accelerator
import os

def train_loop(config,model,noise_scheduler,optimizer,train_dataloader,lr_scheduler):

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),

    )

    if accelerator.is_main_process:

        run = neptune.init_run(
        project="Education/Diffusion-Project",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNGQzNTc4ZS02YjMwLTRhODktOTE3ZC03NzQ3ZThlNzE5ZWMifQ==",
        dependencies = 'infer',
        source_files = ['*.py','*.ipynb']
        
        ) 

        params = asdict(config)

        run['parameters'] = params



        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        accelerator.init_trackers("train_example")

    model,optimizer,train_dataloader,lr_scheduler = accelerator.prepare(
                                                    model,optimizer,train_dataloader,lr_scheduler
                                                )

    global_step = 0

    epoch_loss = 0
    num_batch = len(train_dataloader)
    for epoch in range(config.num_epochs):

        for step,batch in enumerate(train_dataloader):
            clean_images = batch

            noise = torch.randn(clean_images.shape,device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0,noise_scheduler.config.num_train_timesteps, (bs,),device = clean_images.device
                    )
            
            noisy_images = noise_scheduler.add_noise(clean_images,noise,timesteps)

            with accelerator.accumulate(model):

                noise_pred = model(noisy_images,timesteps,return_dict=False)[0]
                loss = torch.nn.MSELoss()(noise_pred,noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            epoch_loss +=logs['loss']

            

            global_step +=1

        if accelerator.is_main_process:

            epoch_loss /= num_batch
            run['train/loss'].append(epoch_loss)
            print(f'Epoch: {epoch}, Epoch loss {epoch_loss}')
                
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet = accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs ==0 or epoch == config.num_epochs-1:
                evaluate(config,epoch,pipeline)

            if (epoch + 1) % config.save_model_epochs ==0 or epoch == config.num_epochs-1:
                pipeline.save_pretrained(config.output_dir)


import neptune
from dataclasses import asdict

def neptune_run():

    run = neptune.init_run(
            project="Education/Diffusion-Project",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNGQzNTc4ZS02YjMwLTRhODktOTE3ZC03NzQ3ZThlNzE5ZWMifQ==",
            dependencies = 'infer',
            source_files = ['*.py','*.ipynb']
            
    ) 

    params = asdict(config)

    run['parameters'] = params

    return run

        

def main(image_path:Path):
    print('Entered Main')
    #run = neptune_run()
    config = TrainingConfig()
    data_loader = load_data(image_path)
    model,optimizer,lr_scheduler = load_train_objs(data_loader)
    noise_scheduler = scheduler()
    train_loop(config,model,noise_scheduler,optimizer,data_loader,lr_scheduler)


if __name__=='__main__':
    
    import sys
    image_path = Path(sys.argv[1])

    if image_path.exists():
        main(image_path)

    else:
        raise FileNotFoundError('Image Path does not exist')



