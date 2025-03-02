# Progressive Growing GAN implementation and training

This project implements Progressive Growing GANs (PGGANs) to generate novel car designs. The model progressively grows both the generator and discriminator, starting from low-resolution images and incrementally adding layers to produce higher-resolution results.
![image](https://github.com/user-attachments/assets/217a1199-4e98-4a77-9839-91bf037f9f54)
Source: Progressive Growing of GANs paper: [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
## Overview

Progressive Growing of GANs is a technique that starts with generating low-resolution images (4x4) and gradually increases the resolution by adding layers to both generator and discriminator networks. This approach provides more stable training and can generate high-quality images up to 128x128 resolution.

## Key Features

- Progressive growing architecture for stable training
- Custom layers implementation (WeightedSum, MinibatchStdev, PixelNormalization)
- Resolution growth from 4x4 to 128x128
- Wasserstein loss with gradient penalty
- Support for GPU training
- Easy inference with pre-trained models

## Requirements

```python
tensorflow>=2.0.0
numpy
Pillow
matplotlib
scikit-image
opencv-python
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vijayparuchuri/ProgressiveAutoGAN.git
cd ProgressiveAutoGAN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your car images in the `data/car_images` directory. The images will be automatically resized to the required dimensions during training.

### Training

```bash
python src/train.py
```

Training parameters can be modified in `train.py`:
- `n_blocks`: Number of growth phases (default: 6)
- `latent_dim`: Size of the latent space (default: 100)
- `n_batch`: Batch sizes for different resolutions
- `n_epochs`: Number of epochs for each resolution

### Inference

To generate new car designs using a trained model:

```bash
python src/inference.py
```

## Model Architecture

The implementation uses a progressive growing architecture with:

- Generator: Starts from 4x4 resolution and progressively grows to 128x128
- Discriminator: Mirror architecture of the generator
- Custom layers:
  - WeightedSum: For smooth transition between resolutions
  - MinibatchStdev: Helps prevent mode collapse
  - PixelNormalization: Normalizes feature vector in each pixel

## Results

The model generates car designs at different resolutions:
- 4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128

Sample outputs can be found in the `checkpoints` directory after training.

## Training Progress

The training process involves:
1. Training initial resolution (4x4)
2. Progressive growing through multiple phases
3. Fade-in of new layers for stable transition
4. Fine-tuning at each resolution

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Progressive Growing of GANs paper: [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
- Car dataset source: [Car Connection Picture Dataset](https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper)
