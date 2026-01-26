# Latent Diffusion Model (LDM) Implementation

## Overview

This implementation provides a complete Latent Diffusion Model pipeline that combines:
1. **VAE Encoder**: Reduces 32³ data to 4³ latent space
2. **DDPM in Latent Space**: Diffusion model operating on the reduced space
3. **VAE Decoder**: Reconstructs 32³ data from latent space

## Quick Start

### 1. Ensure you have a trained VAE

```bash
python train_ae_lim_1.0.py
# Generates: models/vae/ae_epoch=096_val_loss=0.003791.ckpt (or similar)
```

### 2. Train LDM

```bash
python train_ldm.py with encoder_ckpt="models/vae/ae_epoch=096_val_loss=0.003791.ckpt" \
        batch_size=32 \
        lr=0.0001 \
        max_epochs=100 \
        log_dir="logs/ldm"
```

### 3. Generate Samples

```bash
python train_ldm.py with train=False n_sample=20
```


## Files Created/Modified

### Core Implementation Files

#### 1. **DDPM.py** (Modified)
Added the `LDM` class that implements the complete latent diffusion model:

```python
class LDM(pl.LightningModule):
    def __init__(self, _config):
        # Initialize autoencoder (frozen)
        # Initialize DDPM in latent space
        # Setup diffusion process
    
    def encode_to_latent(self, x):  # Data → Latent
    def decode_from_latent(self, z):  # Latent → Data
    def training_step(self, batch, batch_idx):  # Encode → Train DDPM
    def validation_step(self, batch, batch_idx):  # Encode → Validate
```

**Key Features:**
- Automatic encoder/decoder loading from checkpoints
- Frozen encoder/decoder (not trainable)
- Efficient latent space representation (4³ instead of 32³)
- Full integration with PyTorch Lightning

### Training an Inference script

#### 2. **run_ldm.py** (New)
##### Training the LDM model.

**Usage:**
```bash
# Basic training
python train_ldm.py --encoder_ckpt models/vae/ae_epoch=096_val_loss=0.003791.ckpt --train

# With custom parameters
python train_ldm.py with encoder_ckpt="models/vae/ae_epoch=096_val_loss=0.003791.ckpt" \
        batch_size=512 \
        test_only_100=True \
        lr=0.0001 \
        max_epochs=4 \
        log_dir="logs/ldm"
```

**Features:**
- Automatic checkpoint saving
- Early stopping
- TensorBoard logging
- Multi-GPU support (DDP)

##### Sampling with LDM model.
```bash
python train_ldm.py with train=False n_sample=20
```

## Architecture Details

### Data Flow

```
Original Space (32×32×32, 3 channels)
        ↓
    [Encoder] ← Frozen, pre-trained
        ↓
Latent Space (4×4×4, 4 channels)
        ↓
   [DDPM Model] ← Trainable
        ↓
Latent Space (4×4×4, 4 channels)
        ↓
    [Decoder] ← Frozen, pre-trained
        ↓
Original Space (32×32×32, 3 channels)
```

### Latent Space Dimensions

- **Original**: 32³ with 3 channels = 98,304 values per sample
- **Latent**: 4³ with 4 channels = 256 values per sample
- **Compression Ratio**: ~384:1

### DDPM Configuration

The DDPM operates on:
- **Input shape**: (batch_size, 4, 4, 4, 4)
- **Model**: U-Net with `dim=4`
- **Channels**: 4
- **Timesteps**: 1500 (configurable)

## Key Implementation Details

### 1. Encoder/Decoder Management

```python
# In LDM.__init__
from autoencoderldm3d import AutoencoderKL, ddconfig, lossconfig

# Create autoencoder
ae_config = ddconfig(...)
loss_config = lossconfig(...)
self.autoencoder = AutoencoderKL(ae_config, loss_config)

# Load checkpoint
if encoder_ckpt:
    state_dict = torch.load(encoder_ckpt)
    self.autoencoder.load_state_dict(state_dict)

# Freeze
for param in self.autoencoder.parameters():
    param.requires_grad = False
```

### 2. Encoding/Decoding in Training

```python
def training_step(self, batch, batch_idx):
    X, Y = batch
    
    # Encode to latent space
    z = self.encode_to_latent(X)  # (B, 3, 32, 32, 32) → (B, 4, 4, 4, 4)
    
    # Train DDPM on latent
    time = torch.randint(0, self.timesteps, (B,))
    loss = self.diffusion.training_losses(self.model, z, time)
    
    return loss
```

### 3. Sampling Pipeline

```python
	def large_sample_and_decode(self, ldm_model, cell_model, num_sample, directory, target_value=None):	

        # Sample from the latent pace
		latent_samples = self.sample_from_latent(model, grid_size = latent_dim, batch_size = left_over, channels = latent_channels, context = context)
		
        # Decode using the autoencoder
        samples = ldm_model.decode_from_latent(
			latent_samples.detach().to(dtype=torch.float32).to(ldm_model.device)
			)

        # Generate the cell parameter from a previous model
		cell_param_list = cell_model(samples.detach())

```

## Performance Metrics

### Training Efficiency

| Metric | Original DDPM | LDM |
|--------|---------------|-----|
| Data per sample | 98,304 values | 256 values |
| U-Net input size | 32³ | 4³ |
| Memory per batch | ~ ??? | ~ ??? |
| Time per epoch | ~ ??? | ~ ??? |
| Speedup | 1× | ** ???x** |

### Quality vs Speed

- **Original DDPM**: High quality
- **LDM**: quality, efficiency ?
- **Quality Driver**: VAE or DDPM ?

## Configuration

### Important Parameters in `config.py`

```python
# Original space
dim = 32                    # Original spatial dimension
channels = 3               # Original channels

# Latent space (hardcoded in LDM)
z_channels = 4             # Latent channels
latent_dim = 4             # Latent spatial dimension

# Diffusion
timesteps = 1500           # Diffusion steps
loss_type = "huber"        # Loss function

# Training
batch_size = 16
lr = 0.0001
max_epochs = 100
```

## Advanced Usage

### Conditional Generation

```python
# Enable conditional generation
config['self_condition'] = True
config['target_prop'] = 'VF'  # Condition on Void Fraction

# During training, properties are encoded as spatial context
```

### Custom Encoder/Decoder

To use a different VAE:

1. Ensure it has `encode()` and `decode()` methods
2. Verify latent dimensions match the DDPM config
3. Update checkpoint path in config

### Multi-GPU Training

```python
# Automatic with PyTorch Lightning
python train_ldm.py --train

# Uses all available GPUs with DDP strategy
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```bash
python train_ldm.py with batch_size=8 train=True
```

### Issue: Encoder checkpoint not loading

**Check**:
```python
state = torch.load('path/to/checkpoint.ckpt')
print(state.keys())  # Should contain 'state_dict'
```

### Issue: Poor sample quality

**Causes**:
1. Insufficient training → increase `max_epochs`
2. Poor VAE → retrain with better settings
3. Wrong learning rate → try `lr=1e-4` or `5e-5`

### Issue: Validation loss not decreasing

**Solutions**:
1. Check learning rate (try `lr=5e-5`)
2. Verify encoder/decoder are frozen
3. Check that DDPM is operating on correct latent shape

## Comparison: Original DDPM vs LDM

### Original DDPM in run.py

```python
class DDPM(pl.LightningModule):
    def __init__(self, _config):
        # U-Net operates on (B, 3, 32, 32, 32)
        self.model = Unet(dim=32, channels=3, ...)
```

**Problem**: ~100MB per sample in memory

### New LDM

```python
class LDM(pl.LightningModule):
    def __init__(self, _config):
        # Encoder shrinks to (B, 4, 4, 4, 4)
        # U-Net operates on latent space
        self.model = Unet(dim=4, channels=4, ...)
```

**Benefit**: ~1MB per sample + much faster

## References

Implemented based on:
1. Latent Diffusion Models: High-Resolution Image Synthesis with Latent Diffusion Models
2. DDPM: Denoising Diffusion Probabilistic Models
3. AutoencoderKL from PoreGen
