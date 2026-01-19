import os
import json
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
import copy

from dataset import GridDataModule

from Unet import Unet
from Unet_Reg import Unet_Reg
from diffusion import GaussianDiffusion, linear_beta_schedule

import numpy as np

from config import config as _config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap


def accumulate(model1, model2, decay=0.9999):
	par1 = dict(model1.named_parameters())
	par2 = dict(model2.named_parameters())
	
	for k in par1.keys():
		par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def samples_fn(model, diffusion, shape):
	samples = diffusion.p_sample_loop(model=model,
										shape=shape,
										noise_fn=torch.randn)
	return {
		'samples': (samples + 1)/2
	}


def bpd_fn(model, diffusion, x):
	total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = diffusion.calc_bpd_loop(model=model, x_0=x, clip_denoised=True)

	return {
		'total_bpd': total_bpd_b,
		'terms_bpd': terms_bpd_bt,
		'prior_bpd': prior_bpd_b,
		'mse': mse_bt
	}


def validate(val_loader, model, diffusion):
	model.eval()
	bpd = []
	mse = []	
	with torch.no_grad():
		for i, (x, y) in enumerate(iter(val_loader)):
			x       = x
			metrics = bpd_fn(model, diffusion, x)

			bpd.append(metrics['total_bpd'].view(-1, 1))
			mse.append(metrics['mse'].view(-1, 1))

		bpd = torch.cat(bpd, dim=0).mean()
		mse = torch.cat(mse, dim=0).mean()

	return bpd, mse

class DDPM(pl.LightningModule):
	def __init__(self, _config):
		super().__init__()
		self.save_hyperparameters()

		self.config  = _config
		self.grid_size = self.config['dim']
		self.dim_mults = self.config['dim_mults']
		self.channels = self.config['channels']
		self.self_conditioning = self.config['self_condition']
		self.target_prop = self.config['target_prop']
		self.timesteps = self.config['timesteps']
		self.loss_type = self.config['loss_type']
		self.optimizer = self.config['optimizer']	
		self.lr = self.config['lr']


		self.model = Unet(dim = self.grid_size,
						dim_mults = self.dim_mults,
						channels = self.channels,
						self_condition = self.self_conditioning,
						)

		self.ema   = Unet(dim = self.grid_size,
							dim_mults = self.dim_mults,
							channels = self.channels,
							self_condition = self.self_conditioning,
							)

		self.betas = linear_beta_schedule(timesteps = self.timesteps,
										)

		self.diffusion = GaussianDiffusion(betas=self.betas,
											loss_type = self.loss_type,
											)
        
	def forward(self, x):

		return self.diffusion.p_sample_loop(self.model, x.shape)


	def configure_optimizers(self):

		if self.optimizer == 'adam':
			optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		else:
			raise NotImplementedError

		return optimizer

	def training_step(self, batch, batch_idx):

		if self.self_conditioning:
			X,Y = batch
			
			train_batch_size = X.shape[0]

			assert Y.shape[1]==3, 'So far, we only handle following FOUR parameters : VF, HC, HOA'
			assert self.target_prop in ['VF', 'HC', 'HOA'], 'unknown property name is given'

			prop_name = self.target_prop
			if prop_name == 'VF':
				property = Y[:,0]
			elif prop_name == 'HC':
				property = Y[:,1]
			else: # prop_name == 'HOA':
				property = Y[:,2]


			property_context = property.view(train_batch_size, 1, 1, 1, 1).repeat(1, self.channels, self.grid_size, self.grid_size, self.grid_size)


		else: 
			X,_ = batch
			train_batch_size = X.shape[0]


		time = torch.randint(0, self.timesteps, (train_batch_size,)).long().to(X.device)

		if self.self_conditioning:
			loss = self.diffusion.training_losses(self.model, X, time, property_context).mean()

		else:
			loss = self.diffusion.training_losses(self.model, X, time).mean()



		accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 0.9999)

		tensorboard_logs = {'train_loss': loss}

		self.log("test_loss", loss,  sync_dist=True)

		return {'loss': loss, 'log': tensorboard_logs}



	def validation_step(self, batch, batch_idx):

		if self.self_conditioning:
			X,Y = batch
			val_batch_size = X.shape[0]

			assert Y.shape[1]==3, 'So far, we only handle four parameters : VF, HC, HOA'
			assert self.target_prop in ['VF', 'HC', 'HOA'], 'unknown property name is given'

			prop_name = self.target_prop
		
			if prop_name == 'VF':
				property = Y[:,0]
			elif prop_name == 'HC':
				property = Y[:,1]
			else: # prop_name == 'HOA':
				property = Y[:,2]

			property_context = property.view(val_batch_size, 1, 1, 1, 1).repeat(1, self.channels, self.grid_size, self.grid_size, self.grid_size)

		else: 
			X, _ = batch
			val_batch_size = X.shape[0]

		val_batch_size = batch[0].shape[0]
		time = torch.randint(0, self.config['timesteps'], (val_batch_size,)).long().to(X.device)

		if self.self_conditioning:
			loss = self.diffusion.training_losses(self.model, X, time, property_context).mean()

		else:
			loss = self.diffusion.training_losses(self.model, X, time).mean()

		tensorboard_logs = {'valid_loss': loss}

		self.log("val_loss", loss,  sync_dist=True)

		return {'loss': loss, 'log': tensorboard_logs}


class LDM(pl.LightningModule):
	"""
	Latent Diffusion Model that combines:
	1. AutoencoderKL for encoding/decoding
	2. DDPM for diffusion in latent space
	"""
	def __init__(self, _config):
		super().__init__()
		self.save_hyperparameters()

		self.config = _config
		self.grid_size = self.config['dim']
		self.dim_mults = self.config['dim_mults']
		self.channels = self.config['channels']
		self.self_conditioning = self.config['self_condition']
		self.target_prop = self.config['target_prop']
		self.timesteps = self.config['timesteps']
		self.loss_type = self.config['loss_type']
		self.optimizer = self.config['optimizer']	
		self.lr = self.config['lr']
		
		# Load encoder/decoder from config
		self.encoder_ckpt = self.config.get('encoder_ckpt', None)
		self.decoder_ckpt = self.config.get('decoder_ckpt', None)
		
		# Initialize encoder and decoder
		from autoencoderldm3d import AutoencoderKL, ddconfig, lossconfig, ContinuousVAELoss
		
		# Configure encoder/decoder
		ae_config = ddconfig(
			double_z=True,
			z_channels=4,
			resolution=32,
			in_channels=3,
			out_ch=3,
			ch=32,
			ch_mult=[1, 2, 4, 4],
			num_res_blocks=2,
			attn_resolutions=[],
			dropout=0.0,
			has_mid_attn=False
		)
		
		loss_config = lossconfig(
			target=ContinuousVAELoss,
			kl_weight=1e-6
		)
		
		self.autoencoder = AutoencoderKL(ae_config, loss_config)
		
		# Load pretrained encoder/decoder if provided
		if self.encoder_ckpt is not None:
			print(f"Loading encoder from {self.encoder_ckpt}")
			state_dict = torch.load(self.encoder_ckpt, map_location="cpu")["state_dict"] if ".ckpt" in self.encoder_ckpt else torch.load(self.encoder_ckpt, map_location="cpu")
			self.autoencoder.load_state_dict(state_dict, strict=False)
		
		# Freeze encoder/decoder
		for param in self.autoencoder.parameters():
			param.requires_grad = False
		
		# Get latent channels and dimensions
		self.latent_channels = 4  # z_channels from ddconfig
		# Latent spatial dimensions are 1/8 of original (with ch_mult [1,2,4,4])
		self.latent_dim = self.grid_size // 8  # 32 // 8 = 4
		
		# Initialize DDPM in latent space
		self.model = Unet(
			dim=self.latent_dim,
			dim_mults=self.dim_mults,
			channels=self.latent_channels,
			self_condition=self.self_conditioning,
		)
		
		self.ema = Unet(
			dim=self.latent_dim,
			dim_mults=self.dim_mults,
			channels=self.latent_channels,
			self_condition=self.self_conditioning,
		)
		
		# Initialize diffusion in latent space
		self.betas = linear_beta_schedule(timesteps=self.timesteps)
		self.diffusion = GaussianDiffusion(betas=self.betas, loss_type=self.loss_type)
	
	def encode_to_latent(self, x):
		"""Encode data to latent space using the autoencoder."""
		with torch.no_grad():
			z = self.autoencoder.encode(x)
		return z
	
	def decode_from_latent(self, z):
		"""Decode data from latent space using the autoencoder."""
		with torch.no_grad():
			x = self.autoencoder.decode(z)
		return x
	
	def forward(self, x):
		"""Generate samples in original space."""
		# Sample in latent space
		latent_shape = (x.shape[0], self.latent_channels, self.latent_dim, self.latent_dim, self.latent_dim)
		latent_samples = self.diffusion.p_sample_loop(self.model, latent_shape)
		# Decode to original space
		samples = self.decode_from_latent(latent_samples)
		return samples
	
	def configure_optimizers(self):
		if self.optimizer == 'adam':
			optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		else:
			raise NotImplementedError
		return optimizer
	
	def training_step(self, batch, batch_idx):
		if self.self_conditioning:
			X, Y = batch
			train_batch_size = X.shape[0]
			
			assert Y.shape[1] == 3, 'Expected 3 properties: VF, HC, HOA'
			assert self.target_prop in ['VF', 'HC', 'HOA'], 'Unknown property name'
			
			prop_name = self.target_prop
			if prop_name == 'VF':
				property = Y[:, 0]
			elif prop_name == 'HC':
				property = Y[:, 1]
			else:  # HOA
				property = Y[:, 2]
			
			property_context = property.view(train_batch_size, 1, 1, 1, 1).repeat(
				1, self.latent_channels, self.latent_dim, self.latent_dim, self.latent_dim
			)
		else:
			X, _ = batch
			train_batch_size = X.shape[0]
		
		# Encode to latent space
		z = self.encode_to_latent(X)
		
		# Sample random timesteps
		time = torch.randint(0, self.timesteps, (train_batch_size,)).long().to(X.device)
		
		# Compute loss in latent space
		if self.self_conditioning:
			loss = self.diffusion.training_losses(self.model, z, time, property_context).mean()
		else:
			loss = self.diffusion.training_losses(self.model, z, time).mean()
		
		# Update EMA
		accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 0.9999)
		
		tensorboard_logs = {'train_loss': loss}
		self.log("train_loss", loss, sync_dist=True)
		
		return {'loss': loss, 'log': tensorboard_logs}
	
	def validation_step(self, batch, batch_idx):
		if self.self_conditioning:
			X, Y = batch
			val_batch_size = X.shape[0]
			
			assert Y.shape[1] == 3, 'Expected 3 properties: VF, HC, HOA'
			assert self.target_prop in ['VF', 'HC', 'HOA'], 'Unknown property name'
			
			prop_name = self.target_prop
			if prop_name == 'VF':
				property = Y[:, 0]
			elif prop_name == 'HC':
				property = Y[:, 1]
			else:  # HOA
				property = Y[:, 2]
			
			property_context = property.view(val_batch_size, 1, 1, 1, 1).repeat(
				1, self.latent_channels, self.latent_dim, self.latent_dim, self.latent_dim
			)
		else:
			X, _ = batch
			val_batch_size = X.shape[0]
		
		# Encode to latent space
		z = self.encode_to_latent(X)
		
		# Sample random timesteps
		time = torch.randint(0, self.config['timesteps'], (val_batch_size,)).long().to(X.device)
		
		# Compute loss in latent space
		if self.self_conditioning:
			loss = self.diffusion.training_losses(self.model, z, time, property_context).mean()
		else:
			loss = self.diffusion.training_losses(self.model, z, time).mean()
		
		tensorboard_logs = {'valid_loss': loss}
		self.log("val_loss", loss, sync_dist=True)
		
		return {'loss': loss, 'log': tensorboard_logs}
