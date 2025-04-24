# Galaxy Diffusion: Recreating the Cosmos with Generative Models

This repository explores the use of generative diffusion models ([UNet-DDPM](https://arxiv.org/abs/2006.11239), [DiT](https://arxiv.org/abs/2212.09748)) to replicate cosmological simulation data. The primary goal is to assess their potential as fast surrogates for computationally expensive N-body simulations, potentially accelerating prototyping and parameter estimation in cosmology.

## Introduction

Traditional cosmology simulations model the universe's evolution but demand significant computational resources. This project investigates whether deep learning, specifically diffusion models, can generate high-fidelity cosmological data (2D dark matter density fields) much faster while preserving key statistical properties and physical constraints like Periodic Boundary Conditions (PBCs). We evaluate model performance using standard cosmological and image generation metrics.

## Dataset

The models were trained on 2D dark matter density plots derived from the [Quijote simulations](https://github.com/franciscovillaescusa/Quijote-simulations). Specifically, we used 15,000 fiducial ΛCDM simulation realizations, processing the 512x512x512 volumes into 512x512 pixel 2D slices by averaging the first five slices along one axis. Training was performed primarily using data from 8,000 realizations hosted on one cluster.

*Sample of Real Simulation Data:*
![Original Simulation Slice](.\/images/origin.png) 
## Approach and Models

We implemented and evaluated two primary diffusion model architectures:

1.  **UNet-DDPM:** Based on the Denoising Diffusion Probabilistic Model ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) using a U-Net architecture ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)). Code components were adapted from existing implementations (see References).
    *  **Architecture:** U-Net structure with 5 down/up-sampling blocks, incorporating ResNet blocks, Group Normalization, and Attention mechanisms. Total parameters: ~11.22 million.
    *  **Training:** Trained on 512x512 images for 50,000 iterations (batch size 4, gradient accumulation 5) using Adam optimizer (LR 5e-4), MSE loss, sigmoid beta schedule, EMA, and mixed precision.

2.  **DiT (Diffusion Transformer):** A variant ([fastDiT](https://github.com/Shark-Chip/fastDiT)) leveraging the Vision Transformer ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)) backbone, adapted for diffusion ([Peebles & Xie, 2022](https://arxiv.org/abs/2212.09748)).
    *  **Architecture:** Uses a *fixed, pre-trained* autoencoder to work in latent space. Employs a ViT (DiT-B configuration) to process input patches (tested 2x2 and 4x4).
    *  **Training:** Trained using AdamW optimizer (LR 1e-4) and mixed precision. Trained for ~3000 epochs (4x4 patch) or ~1130 epochs (2x2 patch, with early stopping).

*Sample of Generated Simulation Data (UNet-DDPM):*
![Generated Simulation Slice](.\/images/gener.png) <!-- Placeholder - replace with actual image path if available -->

## Evaluation and Results

Models were evaluated using several metrics:

1.  **Power Spectrum (`P(k)`):** Measures structure power across spatial scales (`k`). We use the Mean Absolute Fractional Difference (MAFD) between the generated and real power spectra.
    *  **UNet-DDPM:** Achieved the best MAFD of **2.62%**.  Excels at high-k (small scales).
    *  **DiT Models:** MAFD around 28-29%. Better at low-k (large scales) than UNet-DDPM.
    *  *Comparison:* Our best MAFD significantly outperforms the ~10% reported in related work ([Mudur & Finkbeiner 2022](https://arxiv.org/abs/2211.12444)) on a different dataset (CAMELS).

    ![Power Spectrum Comparison](.\/images/power_spectra_comparison.png)
    ![Power Spectrum Ratio](.\/images/power_spectrum_ratio.png)

2.  **Fréchet Inception Distance (FID):** Standard metric for generative model image quality and realism. Calculated on 5000 generated vs. real samples.
    *  **UNet-DDPM:** Achieved an excellent FID of **0.79** (train) / **0.85** (test).
    *  **DiT Models:** FID scores were significantly higher (~40 train/test).
    *  *Comparison:* The UNet-DDPM FID is state-of-the-art compared to typical diffusion model results on complex datasets like ImageNet (e.g., DiT paper reports 2.27-3.04) or CIFAR-10 (DDPM paper reports 3.17).

3.  **Minkowski Functionals:** Quantify morphology (M0: area, M1: perimeter, M2: Euler characteristic/connectivity).
    *  **UNet-DDPM:** Accurately captured all three functionals (M0, M1, M2).
    *  **DiT Models:** Performed well on M0 and M1 but struggled to accurately reproduce M2, suggesting difficulty with complex topology and boundaries.

4.  **Periodic Boundary Conditions (PBC):** Visually confirmed by tiling generated images.
    *  All models successfully learned to generate images adhering to PBCs *implicitly* from the data, without explicit loss constraints. This is a crucial feature for cosmological applications.

    ![Periodic Boundary Conditions Verification](.\/images/periodic.png)

## Key Findings and Discussion

*  **Model Trade-offs:** UNet-DDPM demonstrates superior overall fidelity (FID, MAFD) and excels at capturing small-scale structures (high-k, M2). DiT models show better performance specifically on large-scale structures (low-k) but lack fine-detail accuracy.
*  **Implicit Constraint Learning:** Diffusion models effectively learn physical constraints like PBCs directly from the training data, a significant advantage for scientific simulations.
*  **State-of-the-Art Fidelity:** The UNet-DDPM achieved exceptionally low FID (0.79/0.85) and MAFD (2.62%) for this complex scientific dataset, indicating high potential for generative models in cosmology.
*  **Computational Efficiency:** Optimal sampling requires ~500-750 timesteps, balancing generation quality and computational cost (though still much faster than full N-body sims).
*  **Limitations:** DiT models struggled with high-frequency details and topological complexity (M2), possibly due to the fixed autoencoder or patch-based approach. UNet-DDPM showed minor deviations at the largest scales (lowest k). FID calculation remains computationally intensive.

<!-- ## Technologies and Libraries

*  **Python:** Core language.
*  **PyTorch:** Deep learning framework.
*  **Hugging Face Libraries:** Likely `diffusers` and concepts from their [Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion).
*  **NumPy:** For numerical operations.
*  **Matplotlib / Seaborn:** For plotting and visualization.
*  **(Potential Cosmology Libraries):** Libraries like `nbodykit` or similar might be used for power spectrum calculation (though not explicitly stated). -->

## Project Structure

```plaintext
.
├── data/          # Input data (processed Quijote slices)
├── models/        # UNet-DDPM and DiT model implementations
├── notebooks/   # Experimentation, analysis, and visualization notebooks
├── results/      # Generated images, model checkpoints, evaluation scores
├── images/       # Supporting images for README, etc.
├── README.md      # This documentation file
└── requirements.txt # Project dependencies (TODO: Add this file)
```

*  **`data/`**: Stores the input 512x512 `.npy` or `.png` slices.
*  **`models/`**: Contains Python scripts defining the UNet and DiT architectures.
*  **`notebooks/`**: Jupyter notebooks for training, inference, evaluation metric calculation, and plotting.
*  **`results/`**: Output directory for model weights, generated samples, metric scores.
*  **`images/`**: Static image files used in documentation.

## Future Work

*  Extend generation to 3D volumes (e.g., 512x512x512).
*  Develop models conditioned on cosmological parameters for efficient emulation and parameter space exploration.
*  Investigate alternative autoencoders or finer patch sizes (e.g., 1x1) for DiT to improve small-scale fidelity.
*  Train models to simulate evolution across redshifts (e.g., from z=127 to z=0).

## References

*  **Quijote Simulations:** Villaescusa-Navarro et al. (2020). *The Quijote Simulations*. [ApJS, 250(1), 2](https://iopscience.iop.org/article/10.3847/1538-4365/aba5a0). ([GitHub](https://github.com/franciscovillaescusa/Quijote-simulations))
*  **DDPM:** Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. [NeurIPS 2020](https://arxiv.org/abs/2006.11239).
*  **DiT:** Peebles, W., & Xie, S. (2022). *Scalable Diffusion Models with Transformers*. [ICCV 2023](https://arxiv.org/abs/2212.09748).
*  **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. [MICCAI 2015](https://arxiv.org/abs/1505.04597).
*  **ViT:** Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [ICLR 2021](https://arxiv.org/abs/2010.11929).
*  **Related Work (CAMELS FID):** Mudur, N., & Finkbeiner, D. P. (2022). *Can denoising diffusion probabilistic models generate realistic astrophysical fields?*. [arXiv:2211.12444](https://arxiv.org/abs/2211.12444).
*  **Hugging Face Diffusion:** [Blog Post](https://huggingface.co/blog/annotated-diffusion), [`diffusers` library](https://huggingface.co/docs/diffusers/index).
*  **Denoising Diffusion Pytorch:** Phil Wang (lucidrains). [GitHub Repository](https://github.com/lucidrains/denoising-diffusion-pytorch). 