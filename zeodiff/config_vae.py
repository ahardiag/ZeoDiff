from sacred import Experiment

ex = Experiment("vae", save_git_info=False)


@ex.config
def config():

	exp_name = "vae"
	train = True
	seed= 42
	batch_size= 8
	max_epochs= 200
	lr= 1e-3
	accelerator= "gpu"
	devices= 1
	n_gpu=1
	precision= 32
	save_dir= "checkpoints_ae"

	# (UNET parameter)
	dim = 32
	dim_mults = (1,2,4)
	channels = 3

	# data
	model_dir = "models/"
	train_dataset = "../data/training/"
	test_dataset = "../data/test/"
	batch_size = 128 # batch size per gpu
	num_workers = 8
	augmentation = True # apply augmentation on database (rotation and translation)
	test_only_100 = False # test trial on dataset of size 100

	# training
	log_dir = "../logs_ae/"
	property_file = "../data/properties.pickle"