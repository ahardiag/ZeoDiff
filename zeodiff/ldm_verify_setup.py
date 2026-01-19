#!/usr/bin/env python3
"""
Quick-start script to verify LDM setup and run the complete pipeline.
Run this script to ensure everything is working correctly.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    dependencies = {
        'torch': torch,
        'pytorch_lightning': None,
        'numpy': None,
    }
    
    for dep_name in dependencies.keys():
        try:
            if dep_name == 'torch':
                print(f"✓ {dep_name} {torch.__version__}")
            else:
                mod = __import__(dep_name)
                print(f"✓ {dep_name} {mod.__version__ if hasattr(mod, '__version__') else 'installed'}")
        except ImportError:
            print(f"✗ {dep_name} NOT FOUND - Install with: pip install {dep_name}")
            return False
    
    return True


def check_files():
    """Check if all required files exist."""
    print("\n" + "="*80)
    print("CHECKING FILES")
    print("="*80)
    
    required_files = {
        'DDPM.py': 'Core DDPM with LDM class',
        'autoencoderldm3d.py': 'VAE encoder/decoder',
        'dataset.py': 'Dataset loading',
        'diffusion.py': 'Diffusion process',
        'config.py': 'Configuration',
        'train_ldm.py': 'LDM training script',
        'ldm_inference.py': 'LDM inference pipeline',
        'ldm_example_pipeline.py': 'Step-by-step example',
        'LDM_README.md': 'LDM documentation',
        'LDM_WORKFLOW.md': 'LDM workflow guide',
    }
    
    missing = []
    for filename, description in required_files.items():
        path = Path(filename)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✓ {filename:30s} ({size_kb:8.1f} KB) - {description}")
        else:
            print(f"✗ {filename:30s} NOT FOUND - {description}")
            missing.append(filename)
    
    if missing:
        print(f"\n✗ Missing {len(missing)} files!")
        return False
    
    return True


def check_encoder_checkpoint(encoder_ckpt):
    """Check if encoder checkpoint exists."""
    print("\n" + "="*80)
    print("CHECKING ENCODER CHECKPOINT")
    print("="*80)
    
    path = Path(encoder_ckpt)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✓ Encoder checkpoint found: {encoder_ckpt}")
        print(f"  Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"✗ Encoder checkpoint NOT FOUND: {encoder_ckpt}")
        print(f"\n  To create an encoder checkpoint:")
        print(f"    python train_ae_lim_1.0.py")
        return False


def test_imports():
    """Test if all modules can be imported."""
    print("\n" + "="*80)
    print("TESTING MODULE IMPORTS")
    print("="*80)
    
    modules = {
        'DDPM.DDPM': 'DDPM class',
        'DDPM.LDM': 'LDM class',
        'autoencoderldm3d.AutoencoderKL': 'AutoencoderKL class',
        'autoencoderldm3d.Encoder': 'Encoder class',
        'autoencoderldm3d.Decoder': 'Decoder class',
        'dataset.GridDataModule': 'GridDataModule class',
        'diffusion.GaussianDiffusion': 'GaussianDiffusion class',
    }
    
    failed = []
    for module_path, description in modules.items():
        try:
            parts = module_path.split('.')
            mod = __import__(parts[0])
            for part in parts[1:]:
                mod = getattr(mod, part)
            print(f"✓ {module_path:35s} - {description}")
        except Exception as e:
            print(f"✗ {module_path:35s} - ERROR: {str(e)[:50]}")
            failed.append(module_path)
    
    if failed:
        print(f"\n✗ Failed to import {len(failed)} modules")
        return False
    
    return True


def test_ldm_initialization(encoder_ckpt):
    """Test if LDM model can be initialized."""
    print("\n" + "="*80)
    print("TESTING LDM INITIALIZATION")
    print("="*80)
    
    try:
        from config import config as _config
        from DDPM import LDM
        
        print("✓ Imports successful")
        
        # Create config
        config = _config()
        config['encoder_ckpt'] = encoder_ckpt
        config['decoder_ckpt'] = encoder_ckpt
        config['train'] = True
        
        print("✓ Config created")
        
        # Initialize LDM
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ldm = LDM(config).to(device)
        
        print("✓ LDM model initialized")
        print(f"  - Device: {device}")
        print(f"  - Latent dimension: {ldm.latent_dim}³")
        print(f"  - Latent channels: {ldm.latent_channels}")
        print(f"  - Timesteps: {ldm.timesteps}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32, 32).to(device)
        
        # Test encoding
        z = ldm.encode_to_latent(dummy_input)
        print(f"✓ Encoding works: {dummy_input.shape} → {z.shape}")
        
        # Test decoding
        x_recon = ldm.decode_from_latent(z)
        print(f"✓ Decoding works: {z.shape} → {x_recon.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ LDM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_quick_start_guide():
    """Print quick start guide."""
    print("\n" + "="*80)
    print("QUICK START GUIDE")
    print("="*80)
    
    guide = """
1. TRAIN LDM:
python train_ldm.py with encoder_ckpt="models/vae/ae_epoch=096_val_loss=0.003791.ckpt" \\
        batch_size=512 \\
        test_only_100=True \\
        lr=0.0001 \\
        max_epochs=4 \\
        log_dir="logs/ldm"
        
2. GENERATE SAMPLES:
   python -c "
   from ldm_inference import LDMInferencePipeline
   import torch
   
   pipeline = LDMInferencePipeline(
       encoder_ckpt='models/vae/ae_epoch=096_val_loss=0.003791.ckpt',
       ddpm_ckpt='logs/ldm_training/last.ckpt',
       device='cuda'
   )
   
   samples = pipeline.sample(num_samples=10, batch_size=2)
   torch.save(samples, 'samples.pt')
   "

3. RUN EXAMPLE PIPELINE:
   python ldm_example_pipeline.py

4. VIEW DOCUMENTATION:
   - LDM_README.md: Implementation details
   - LDM_WORKFLOW.md: Complete workflow guide

5. MONITOR TRAINING:
   tensorboard --logdir=logs/ldm_training
"""
    
    print(guide)


def main():
    """Run all checks."""
    parser = argparse.ArgumentParser(description='Verify LDM setup')
    parser.add_argument('--encoder_ckpt', type=str, 
                       default='models/vae/ae_epoch=096_val_loss=0.003791.ckpt',
                       help='Path to encoder checkpoint')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LATENT DIFFUSION MODEL - SETUP VERIFICATION")
    print("="*80)
    
    all_passed = True
    
    # Check dependencies
    if not check_dependencies():
        all_passed = False
    
    # Check files
    if not check_files():
        all_passed = False
    
    # Check encoder checkpoint
    encoder_exists = check_encoder_checkpoint(args.encoder_ckpt)
    if not encoder_exists:
        all_passed = False
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test LDM initialization
    if encoder_exists and not test_ldm_initialization(args.encoder_ckpt):
        all_passed = False
    
    # Print results
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - SYSTEM READY!")
        print_quick_start_guide()
        return 0
    else:
        print("✗ SOME CHECKS FAILED - SEE ABOVE FOR DETAILS")
        print("\nTroubleshooting:")
        print("  1. Install missing dependencies")
        print("  2. Check file paths")
        print("  3. Train VAE encoder if missing")
        print("  4. Review error messages above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
