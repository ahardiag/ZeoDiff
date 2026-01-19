"""
Latent Diffusion Model Training and Inference Pipeline

This script demonstrates the complete pipeline:
1. Load dataset
2. Load pre-trained VAE encoder/decoder
3. Train DDPM in latent space
4. Generate samples and decode back to original space
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
import copy

from dataset import GridDataModule
from DDPM import LDM
from config import config as _config
from config import ex

@ex.automain
def main(_config):
	"""
	Main training function for LDM.
	"""
	
	# Set seed
	pl.seed_everything(_config["seed"])
	_config = copy.deepcopy(_config)
	
	# Create model
	exp_name = f"{_config['exp_name']}_latent"
	ldm_model = LDM(_config)
	
	# Create data module
	grid_data_module = GridDataModule(_config)
	
	# Setup logger
	logger = pl.loggers.tensorboard.TensorBoardLogger(
		_config["log_dir"],
		name=f'{exp_name}_seed_{_config["seed"]}_train_True_batchsize_{_config["batch_size"]}_'
		     f'target_{_config["target_prop"]}_lr_{_config["lr"]}_timesteps_{_config["timesteps"]}_'
		     f'loss_{_config["loss_type"]}',
	)
	
	# Training
	if _config.get("train", True):
		
		if _config.get("load_model") is not None:
			load_model_dir = os.path.join(_config['model_dir'], _config['load_model'])
			state_dict = torch.load(load_model_dir)
			ldm_model.load_state_dict(state_dict['state_dict'])
			print(f"Loaded model from {load_model_dir}")
		
		os.makedirs(_config["log_dir"], exist_ok=True)
		
		checkpoint_callback = ModelCheckpoint(
			filename=os.path.join(_config['save_dir'], 'ldm_{epoch:02d}-{val_loss:.6f}'),
			monitor='val_loss',
			verbose=True,
			save_last=True,
			save_top_k=1,
			mode='min',
		)
		
		early_stopping_callback = EarlyStopping(
			monitor='val_loss',
			patience=_config.get('early_stopping', 10),
			verbose=True
		)
		
		trainer = pl.Trainer(
			accelerator=_config['accelerator'],
			devices=_config['devices'],
			num_nodes=_config.get("num_nodes", 1),
			max_epochs=_config.get('max_epochs', 100),
			precision=_config.get('precision', 16),
			callbacks=[checkpoint_callback, early_stopping_callback],
			logger=logger,
			strategy=DDPStrategy(find_unused_parameters=True),
		)
		
		print("\n" + "="*80)
		print("Starting Latent Diffusion Model Training")
		print("="*80)
		print(f"Encoder checkpoint: {_config.get('encoder_ckpt', 'None')}")
		print(f"Decoder checkpoint: {_config.get('decoder_ckpt', 'None')}")
		print(f"Latent space dimensions: {ldm_model.latent_dim}x{ldm_model.latent_dim}x{ldm_model.latent_dim}")
		print(f"Latent channels: {ldm_model.latent_channels}")
		print("="*80 + "\n")
		
		trainer.fit(ldm_model, grid_data_module)
	
	# Sampling
	else:
		ldm_model.cuda()
		
		eval_model = _config.get('eval_model')
		if eval_model is not None:
			load_model_dir = os.path.join(_config['model_dir'], eval_model)
			state_dict = torch.load(load_model_dir)
			ldm_model.load_state_dict(state_dict['state_dict'])
			print(f"Loaded model from {load_model_dir}")
		
		ldm_model.eval()
		
		if not os.path.exists(_config.get('sample_dir', 'samples/')):
			os.makedirs(_config.get('sample_dir', 'samples/'), exist_ok=True)
		
		print("\n" + "="*80)
		print("Starting Latent Diffusion Model Sampling")
		print("="*80)
		print(f"Number of samples: {_config.get('n_sample', 10)}")
		print(f"Sample directory: {_config.get('sample_dir', 'samples/')}")
		print("="*80 + "\n")
		
		# Generate samples
		n_samples = _config.get('n_sample', 10)
		batch_size = min(n_samples, _config['batch_size'])
		num_batches = (n_samples + batch_size - 1) // batch_size
		
		all_samples = []
		with torch.no_grad():
			for batch_idx in range(num_batches):
				print(f"Generating batch {batch_idx + 1}/{num_batches}...")
				samples = ldm_model(torch.randn(batch_size, 3, 32, 32, 32).cuda())
				all_samples.append(samples.cpu())
		
		all_samples = torch.cat(all_samples, dim=0)[:n_samples]
		
		# Save samples
		sample_path = os.path.join(_config.get('sample_dir', 'samples/'), 'ldm_samples.pt')
		torch.save(all_samples, sample_path)
		print(f"Saved {n_samples} samples to {sample_path}")
		print(f"Sample shape: {all_samples.shape}")
