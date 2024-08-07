from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader,Dataset
import pathlib
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
import neptune as neptune
import argparse


from evaluate import sample_images
from customDataClass import CosmosImageData
from model import unet_model
from train import train_loop

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True,help='Enter the image paths directly')
args = parser.parse_args()

image_path = Path(args.data_path)


NEPTUNE_API_KEY = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNGQzNTc4ZS02YjMwLTRhODktOTE3ZC03NzQ3ZThlNzE5ZWMifQ=='

run = neptune.init_run(
    project="Education/cosmos",
    api_token=NEPTUNE_API_KEY,
    dependencies='infer',
    source_files=['*.py']
)

from config import TrainingConfig
from dataclasses import asdict


config = TrainingConfig()

run["parameters"] = asdict(config)

image_transform = transforms.Compose([
        transforms.Resize((config.image_size,config.image_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.5],[0.5])

])

dataset = CosmosImageData(image_path,transform = image_transform)
train_dataloader  = DataLoader(dataset,batch_size = config.train_batch_size,shuffle=True)



model = unet_model(config)

sample_image = dataset[0].unsqueeze(0)
print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)



noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule ='squaredcos_cap_v2')
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([250])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler,run)

train_loop(*args)