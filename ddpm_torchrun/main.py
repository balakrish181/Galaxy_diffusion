import torch
from diffusers import DDPMScheduler
from config import TrainingConfig
import neptune
import os
import sys 
from dataclasses import asdict
from customDataClass import CosmosImageData
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image



config = TrainingConfig()

class Trainer(TrainingConfig):
    def __init__(self) -> None:
        print(self.learning_rate)
        self.run = self.init_neptune()

        pass

    def init_neptune(self):
        max_retries = 2
        attempt = 0
        run = None
        
        while attempt <= max_retries:
            if os.environ.get('NEPTUNE_API_TOKEN'):
                NEPTUNE_API_KEY = os.environ.get('NEPTUNE_API_TOKEN')
            else:
                NEPTUNE_API_KEY = input('Enter Neptune Key: ')

            try:
                run = neptune.init_run(
                    project="Education/cosmos",
                    api_token=NEPTUNE_API_KEY            
                )
                break  # Break out of the loop if successful

            except Exception as e:
                print(f"Failed to initialize Neptune run. Attempt {attempt + 1} of {max_retries + 1}. Error: {e}")
                attempt += 1

                if attempt > max_retries:
                    print("Max retries reached. Exiting.")
                    sys.exit(1)

        return run
        
    def params_upload_neptune(self):
        self.run['parameters'] = asdict(TrainingConfig())

    def train():
        pass

    def evaluate():
        pass


def prepare_dataloader(image_path:str,batch_size:int = 16):
    image_transform = transforms.Compose([
        transforms.Resize((config.image_size)),
        transforms.ToTensor()
    ])

    dataset = CosmosImageData(image_path,transform=image_transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler = DistributedSampler(dataset)
    )

    return data_loader

def load_train_objs(train_dataloader):
    model = UNet2DModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
            )

    return model,optimizer,lr_scheduler

def working_status(dataloader):
    sample_image = next(iter(dataloader))[0].unsqueeze(0)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule ='squaredcos_cap_v2')
    noise = torch.randn(sample_image.shape)
    timesteps = torch.LongTensor([250])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

    Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

    pass


if __name__=='__main__':
    cls = Trainer()
