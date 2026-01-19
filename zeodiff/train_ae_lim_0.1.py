import os
import copy
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

from dataset import GridDataModule
from autoencoderldm3d import ContinuousVAELoss,AutoencoderKL, ddconfig, lossconfig

from config_vae import config as _config
from config_vae import ex

# --------------------------------------------------
# Main training function
# --------------------------------------------------

def run(log_dir="logs_ae/", train=True, **kwargs):

    config = _config()
    for key in kwargs.keys():
        assert key in config, 'wrong config arguments are given as an input.'

    config.update(kwargs)
    config["log_dir"] = log_dir
    config["train"] = train.lower()=='true'

    main(config)

@ex.automain
def main(_config):

    pl.seed_everything(_config["seed"])

    _config = copy.deepcopy(_config)

    os.makedirs(_config["save_dir"], exist_ok=True)

    # -----------------------------
    # Model configuration
    # -----------------------------

    ae_ddconfig = ddconfig(
        double_z=True,
        z_channels=4,
        resolution=32,
        in_channels=3,      # 3 channels : Si + O + energy 
        out_ch=3,
        ch=32,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        dropout=0.0,
        has_mid_attn=True,
    )

    ae_lossconfig = lossconfig(
        target=ContinuousVAELoss,   # your MSE+KL loss
        kl_weight=1e-6,
    )

    model = AutoencoderKL(
        ddconfig=ae_ddconfig,
        lossconfig=ae_lossconfig,
        embed_dim=4,
    )

    model.set_optimizer_and_scheduler(
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=_config["lr"],
            weight_decay=1e-4,
        )
    )

    # -----------------------------
    # Data
    # -----------------------------

    datamodule = GridDataModule(_config)

    # -----------------------------
    # Logging & callbacks
    # -----------------------------

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=_config["exp_name"],
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=_config["save_dir"],
        filename="ae_{epoch:03d}_{val_loss:.6f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=30,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=_config["devices"],
        precision=_config["precision"],
        max_epochs=_config["max_epochs"],
        callbacks=[checkpoint_cb, early_stop_cb,TQDMProgressBar(refresh_rate=100)],
        logger=logger,
        limit_train_batches=0.1,      # TEST
        limit_val_batches=0.1,        # TEST
    )
    
    trainer.fit(model, datamodule)
