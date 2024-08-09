import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm.auto import tqdm
from diffusers import DDPMPipeline, DDPMScheduler
from evaluate import sample_images
from model import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import neptune
import sys
from torchvision import transforms

from config import TrainingConfig
from customDataClass import CosmosImageData
from pathlib import Path

config = TrainingConfig()

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, train_data, config, noise_scheduler):
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.gradient_accumulation_steps = 4  # Accumulate gradients over 4 steps

        self.config = config
        self.train_data = train_data
        self.run = self.init_neptune()
        self.epochs_run = 0
        self.pipeline = None  # TODO: Add a pipeline for inference
        self.model = model
        self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.noise_scheduler = noise_scheduler

    def init_neptune(self):
        max_retries = 2
        attempt = 0
        run = None

        while attempt <= max_retries:
            # if os.environ.get('NEPTUNE_API_TOKEN'):
            #     NEPTUNE_API_KEY = os.environ.get('NEPTUNE_API_TOKEN')
            
            if True:
                NEPTUNE_API_KEY = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNGQzNTc4ZS02YjMwLTRhODktOTE3ZC03NzQ3ZThlNzE5ZWMifQ=='
            
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

    def train_loop(self, max_epochs):
        self.model.train()
        for epoch in range(max_epochs):
            self.train_data.sampler.set_epoch(epoch)
            epoch_loss = 0
            progress_bar = tqdm(total=len(self.train_data), disable=self.gpu_id != 0)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(self.train_data):

                print(f'Step : {step}')
                clean_images = batch.to(self.gpu_id)
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                    dtype=torch.int64
                )
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss = loss / self.gradient_accumulation_steps  # Scale loss

                loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                if self.gpu_id == 0:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), lr=self.lr_scheduler.get_last_lr()[0])

            avg_epoch_loss = epoch_loss / len(self.train_data)
            print(f"Epoch {epoch} - Average Loss: {avg_epoch_loss:.4f}")

            if self.gpu_id == 0:
                self.run['train/loss'].append(avg_epoch_loss)
                self.run['learning_rate'].append(self.lr_scheduler.get_last_lr()[0])

                if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
                    sample_images(self.config, epoch, self.pipeline)

                if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
                    pipeline = DDPMPipeline(unet=self.model.module, scheduler=self.noise_scheduler)
                    pipeline.save_pretrained(self.config.output_dir)



    # def evaluate(self):
    #     sample_images(self.config, self.epochs_run, self.pipeline)


def ddp_setup():
    init_process_group(backend='nccl')


def prepare_dataloader(image_path: str, batch_size: int = 16, config=None):
    image_transform = transforms.Compose([
        transforms.Resize((config.image_size)),
        transforms.ToTensor()
    ])

    dataset = CosmosImageData(image_path, transform=image_transform)

    print(f'Length of dataset : {len(dataset)}')
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    print(f'Length of dataloader : {len(data_loader)}')
    return data_loader


def load_train_objs(train_dataloader, config):
    model = UNet2DModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    return model, optimizer, lr_scheduler


def main(image_path: str, total_epochs: int = config.num_epochs, save_every: int = config.save_image_epochs, batch_size: int = 4):
    config = TrainingConfig()
    ddp_setup()

    train_data = prepare_dataloader(image_path, batch_size, config)

    model, optimizer, lr_scheduler = load_train_objs(train_data, config)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    trainer = Trainer(model, optimizer, lr_scheduler, train_data, config, noise_scheduler)
    trainer.train_loop(total_epochs)

    destroy_process_group()


if __name__ == '__main__':
    import sys

    image_path = sys.argv[1]
    image_path = Path(image_path)
    print(image_path)
    main(image_path=image_path)
