
from PIL import Image
from pathlib import Path
import torch 
from torchvision import transforms

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet,GaussianDiffusion,Trainer


paths = Path('../images2.0/')
print(paths.exists())

image_paths = list(Path(paths).glob('*.npy'))

#changed here

def main():
    model = Unet(
    32,
    dim_mults= (1,2,4,4,8),
    flash_attn=True
)
    print(f'Number of parameters in the model is {sum(i.numel() for i in model.parameters())}')


    diffusion = GaussianDiffusion(
        model,
        image_size=512,
        timesteps=1000,
        sampling_timesteps=250
    )

    trainer = Trainer(
        diffusion,
        paths,
        train_batch_size = 4,
        train_lr = 5e-4,
        train_num_steps = 60000,         # total training steps
        gradient_accumulate_every = 5,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
        save_and_sample_every = 2000,
        num_samples = 4,
        results_folder = './results',
    
    )
    #trainer.load(3)
    trainer.train()

if __name__ =='__main__':
    main()