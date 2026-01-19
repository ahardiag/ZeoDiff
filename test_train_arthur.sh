# Train the Autoencoder (VAE) model
cd zeodiff
python train_ae.py with train_dataset='../data/train/' \
    test_dataset='../data/test/' \
    max_epochs=50 \
    n_gpu=1 \
    devices=1 \
    batch_size=32

# Train the original DDPM model from Park
cd zeodiff
python train_ddpm.py with train=True \
    train_dataset='../data/train/' \
    test_dataset='../data/test/' \
    logs_dir='logs/ddpm' \
    model_dir='models/ddpm' \
    target_prop='unconditional' \
    max_epochs=100 \
    n_gpu=1 \
    devices=1 \
    batch_size=32

# Train the LDM model
python train_ldm.py with encoder_ckpt="models/vae/ae_epoch=096_val_loss=0.003791.ckpt" \
        batch_size=32 \
        lr=0.0001 \
        max_epochs=100 \
        log_dir="logs/ldm"