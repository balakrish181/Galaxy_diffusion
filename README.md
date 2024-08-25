# Cosmology Simulations with Generative AI

This repository explores the use of Generative AI models to replicate cosmology simulation data, aiming to significantly reduce the time required for prototyping and parameter estimation.

## Introduction

Traditionally, cosmology simulations can take hours, or even thousands of core hours, to run. Our goal is to reduce this prototyping time by leveraging Generative AI to create data that closely resembles the output of traditional simulations. By maintaining the underlying constraints and structures of the simulations, we aim to provide a faster and more efficient method for estimating cosmological parameters.

## Approach

To achieve our goal, we explored different Generative AI models, including:

- **DiT (Diffusion Transformers)**: Effective in capturing complex data distributions but may require more data for optimal performance.
- **DDPM (Denoising Diffusion Probabilistic Models)**: A generative model that leverages diffusion processes to generate high-quality data. It performed exceptionally well with our smaller dataset of approximately 9000 samples.

While the Transformer models (DiT) showed good data representation, the UNet model demonstrated superior performance for our dataset size, providing more accurate and reliable generated data.

## Evaluation

To evaluate the quality of the generated data, we used power spectrum analysis to compare it with the original simulated data. The evaluation metric employed was the Mean Absolute Fractional Difference (MAFD), which measures the discrepancy between the generated and actual data. Our results yielded an MAFD of 1.10%, indicating high fidelity in the generated simulations.




