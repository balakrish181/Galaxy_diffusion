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
from evaluate import sample_images
from torch.distributed import init_process_group,destroy_process_group
from tqdm.auto import tqdm
from diffusers import DDPMPipeline,DDPMScheduler




config = TrainingConfig()

def ddp_setup():
    init_process_group(backend='nccl')

class Trainer(TrainingConfig):
    def __init__(self,model,optimizer,lr_scheduler,train_data) -> None:
        print(self.learning_rate)
        self.train_data = train_data
        self.config = TrainingConfig()
        self.run = self.init_neptune()
        self.epochs_run = 0
        self.pipeline = None   # TODO: Add a pipeline for inference
        self.model = model
        self.optimizer = optimizer
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.lr_scheduler = lr_scheduler
        self.model = DDP(self.model,device_ids = [self.gpu_id])
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule ='squaredcos_cap_v2')

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

    def train_loop(self,max_epochs):

        for epoch in range(max_epochs):
            
            epoch_loss = 0

            for step,batch in enumerate(self.train_data):
                clean_images = batch
                noise = torch.randn(clean_images.shape,device = clean_images.device)
                bs = clean_images.shape[0]

                timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                        dtype=torch.int64
                        )
                
                noisy_image = self.noise_scheduler.add_noise(clean_images,noise,timesteps)
                




        pass

    def evaluate(self):
        sample_images(self.config,self.epochs_run,self.pipeline)

    def load_snapshot():
        pass

    def save_snapshot():
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

    print(Image)
    print('Scheduler success')


def main(image_path:str,total_epochs:int,save_every:int,batch_size:int=16,snapshot_path:str = 'snapshot.pth'):
    ddp_setup()
    train_data = prepare_dataloader(image_path,batch_size)
    model,optimizer,lr_scheduler = load_train_objs(train_data)
    working_status(train_data)
    trainer = Trainer(model,optimizer,lr_scheduler)




if __name__=='__main__':
    cls = Trainer()
