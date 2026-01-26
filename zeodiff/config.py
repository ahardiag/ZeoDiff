from sacred import Experiment

ex = Experiment("ldm", save_git_info=False)

@ex.config
def config():

	exp_name = "ldm"

	seed = 42
	train = True
	precision = 16

	# ddpm model
	# (UNET parameter)
	dim = 32
	dim_mults = (1,2,4) # 32x32x32 -> (1,2,4) to keep low for small images
	channels = 3 
	self_condition = False
	target_prop = None
	# (Diffusion parameter)
	timesteps = 1500
	loss_type = "huber"

	# autoencoder (VAE) for latent space
	encoder_ckpt = "models/vae/ae_epoch=096_val_loss=0.003791.ckpt"
	decoder_ckpt = "models/vae/ae_epoch=096_val_loss=0.003791.ckpt"

	# cell parameter prediction model
	c_model_dir = "models/lattice_regressor.ckpt"
	c_dim_mults = (1,2,4)

	# data
	model_dir = "models/"
	train_dataset = "../data/train/"
	test_dataset = "../data/test/"
	batch_size = 64 # batch size per gpu
	num_workers = 8
	augmentation = True # apply augmentation on database (rotation and translation)
	test_only_100 = False # test trial on dataset of size 100
	property_file = "../data/properties.pickle"


	# training
	accelerator = "gpu"
	n_gpu = 1
	devices = 1
	num_nodes = 1
	optimizer = "adam"
	lr = 1e-4
	log_dir = "../logs/ldm/"
	max_epochs = 10
	n_iter = 10000000
	save_dir = "models/ldm" # where models designated by callbacks will be stored
	grid_size = 32
	strategy = "ddp" # DDPStrategy(find_unused_parameters=True) is now being used as default. If you want to change it, modify trainer part of run.py.
	early_stopping = 50
	load_model = None


	# sampling / evaluation
	eval_model = "ldm/ldm_epoch=27-val_loss=0.106627.ckpt"
	#eval_model = "unconditional.ckpt"
	sample_dir = "../samples/"
	sample_freq = 200
	target_value = 0.05 # do not use None for unconditional sampling
						# instead use self_condition = False
	n_sample = 2
