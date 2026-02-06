#!/usr/bin/env python
"""
Hyperparameter Sweep Script for SAE Training on Nucleotide Transformer

This script trains multiple SAEs across a range of hyperparameters on a selected model, layer, and dataset.
Supports comprehensive hyperparameter sweeps including:
- Learning rates
- L1 coefficients (sparsity penalties)
- Dictionary expansion ratios
- Batch sizes
- Training steps

Usage:
    python scripts/train_sae_hyperparameter_sweep.py --model 50 --layer 6 --data-dir /path/to/data

    # Or with environment variables:
    export INTERPLM_DATA=/path/to/data
    python scripts/train_sae_hyperparameter_sweep.py --model 50 --layer 6

Command Line Arguments:
    --model: Model size (50, 100, 250, 500, 2500 for respective NT models)
    --layer: Layer number to train SAE on (e.g., 6)
    --data-dir: Path to data directory containing training embeddings
    --max-combinations: Limit to random subset of combinations (optional)
    --learning-rates: Custom learning rates (optional, space-separated)
    --l1-coefficients: Custom L1 coefficients (optional, space-separated)
    --expansion-ratios: Custom expansion ratios (optional, space-separated)
    --batch-sizes: Custom batch sizes (optional, space-separated)
    --training-steps: Custom training steps (optional, space-separated)
    --use-wandb: Enable W&B logging
    --wandb-entity: W&B entity name (required if using W&B)
"""

import argparse
import gc
import itertools
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Set PyTorch CUDA allocator configuration to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from interplm_training.checkpoint_manager import CheckpointManager
from interplm_training.configs import (
    CheckpointConfig,
    DataloaderConfig,
    TrainingRunConfig,
    WandbConfig,
)
from interplm_training.nt_activation_buffer import (
    NTActivationBuffer,
)
from interplm_training.nt_fidelity import NTFidelityConfig
from interplm_training.trainers.batch_top_k import (
    BatchTopKTrainer,
    BatchTopKTrainerConfig,
)
from interplm_training.training_run import SAETrainingRun
from interplm_training.wandb_manager import WandbManager

# Hyperparameter ranges to sweep over
HYPERPARAMETER_GRID = {
    "learning_rates": [3e-4],  # Fixed learning rate
    "ks": [16, 32, 64],  # Number of active features per batch
    "auxk_alphas": [1 / 32],  # Fixed auxiliary loss weight
    "threshold_betas": [0.999],  # Fixed threshold update rate
    "expansion_ratios": [4, 8, 16, 32],  # Expansion factors
    "batch_sizes": [4096],  # Fixed batch size
    "training_steps": [
        24_414
    ],  # Approximately 1B tokens (1B / 4096 / 10 tokens per sample)
}

# Small test grid for debugging
TEST_HYPERPARAMETER_GRID = {
    "learning_rates": [3e-4],
    "ks": [32],  # Single k value
    "auxk_alphas": [1 / 32],
    "threshold_betas": [0.999],
    "expansion_ratios": [8],  # Single expansion ratio
    "batch_sizes": [512],  # Smaller batch size
    "training_steps": [100],  # Much shorter training
}

# Model configurations - maps model size to config
MODEL_CONFIGS = {
    50: {
        "name": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "short": "nt-50m",
        "dim": 512,
    },
    100: {
        "name": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "short": "nt-100m",
        "dim": 512,
    },
    250: {
        "name": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "short": "nt-250m",
        "dim": 768,
    },
    500: {
        "name": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "short": "nt-500m",
        "dim": 1024,
    },
    2500: {
        "name": "InstaDeepAI/nucleotide-transformer-v2-2.5b-multi-species",
        "short": "nt-2.5b",
        "dim": 2560,
    },
}


def get_gpu_memory_gb(device_id: int) -> float:
    """Get available GPU memory in GB"""
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        torch.cuda.set_device(device_id)
        return torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    return 0.0


def cleanup_gpu_memory():
    """Comprehensive GPU memory cleanup between training runs"""
    if torch.cuda.is_available():
        # Force garbage collection
        gc.collect()

        # Clear PyTorch cache
        torch.cuda.empty_cache()

        # Force synchronization to ensure cleanup completes
        torch.cuda.synchronize()

        print("🧹 GPU memory cleanup completed")


def get_current_gpu_usage():
    """Get current GPU memory usage for monitoring"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0.0, 0.0


def filter_configs_by_memory(
    configs: List[Dict], min_memory_gb: float = 8.0
) -> List[Dict]:
    """Filter configurations based on estimated memory requirements"""
    filtered = []

    for config in configs:
        # Rough memory estimation based on model size and batch size
        embedding_dim = config["embedding_dim"]
        dictionary_size = config["dictionary_size"]
        batch_size = config["batch_size"]

        # Estimate memory usage (very rough approximation)
        model_params = embedding_dim * dictionary_size * 2  # encoder + decoder
        activation_memory = batch_size * embedding_dim * dictionary_size
        estimated_gb = (
            (model_params + activation_memory) * 4 / (1024**3)
        )  # 4 bytes per float32

        if estimated_gb <= min_memory_gb * 0.8:  # Leave some headroom
            filtered.append(config)
        else:
            print(
                f"Skipping config due to memory constraints: {config['run_name']} (estimated {estimated_gb:.1f}GB)"
            )

    return filtered


def generate_hyperparameter_combinations(
    model_short: str,
    embedding_dim: int,
    layer: int,
    data_dir: Path,
    hyperparameter_grid: Dict[str, List],
    max_combinations: int = None,
) -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters"""

    # Create all combinations
    combinations = list(
        itertools.product(
            hyperparameter_grid["learning_rates"],
            hyperparameter_grid["ks"],
            hyperparameter_grid["auxk_alphas"],
            hyperparameter_grid["threshold_betas"],
            hyperparameter_grid["expansion_ratios"],
            hyperparameter_grid["batch_sizes"],
            hyperparameter_grid["training_steps"],
        )
    )

    # Limit combinations if specified
    if max_combinations and len(combinations) > max_combinations:
        random.shuffle(combinations)
        combinations = combinations[:max_combinations]
        print(
            f"Limited to {max_combinations} random combinations out of {len(list(itertools.product(*hyperparameter_grid.values())))}"
        )

    configs = []
    for i, (
        lr,
        k,
        auxk_alpha,
        threshold_beta,
        exp_ratio,
        batch_size,
        steps,
    ) in enumerate(combinations):
        dictionary_size = embedding_dim * exp_ratio

        config = {
            "run_id": i,
            "run_name": f"{model_short}_L{layer}_lr{lr:.0e}_k{k}_auxk{auxk_alpha:.3f}_tb{threshold_beta:.4f}_exp{exp_ratio}_bs{batch_size}_steps{steps}",
            "model_short": model_short,
            "layer": layer,
            "embedding_dim": embedding_dim,
            "dictionary_size": dictionary_size,
            "learning_rate": lr,
            "k": k,
            "auxk_alpha": auxk_alpha,
            "threshold_beta": threshold_beta,
            "expansion_ratio": exp_ratio,
            "batch_size": batch_size,
            "training_steps": steps,
            "data_dir": data_dir,
        }
        configs.append(config)

    return configs


def create_training_config(
    hp_config: Dict[str, Any],
    model_name: str,
    save_base_dir: Path,
    eval_seq_file: Path = None,
    use_wandb: bool = False,
    data_mode: str = "embeddings",  # "embeddings" or "sequences"
    sequences_file: Path = None,
    finetuned_path: Path = None,
) -> TrainingRunConfig:
    """Create a training configuration from hyperparameters"""

    save_dir = save_base_dir / hp_config["run_name"]

    # Create dataloader config based on data mode
    if data_mode == "embeddings":
        # Pre-saved embeddings mode
        embeddings_dir = (
            hp_config["data_dir"]
            / "training_embeddings"
            / hp_config["model_short"]
            / f"layer_{hp_config['layer']}"
        )
        dataloader_cfg = DataloaderConfig(
            plm_embd_dir=embeddings_dir,
            batch_size=hp_config["batch_size"],
            seed=42,  # Fixed seed for reproducibility
        )
    else:
        # On-the-fly sequences mode - we'll handle this differently
        # For now, create a placeholder dataloader config
        dataloader_cfg = None

    # Calculate warmup and decay steps
    warmup_steps = max(1, min(100, int(hp_config["training_steps"] * 0.1)))
    decay_start = max(warmup_steps + 1, int(hp_config["training_steps"] * 0.8))

    # Trainer config
    trainer_cfg = BatchTopKTrainerConfig(
        activation_dim=hp_config["embedding_dim"],
        dictionary_size=hp_config["dictionary_size"],
        lr=hp_config["learning_rate"],
        k=hp_config["k"],
        auxk_alpha=hp_config["auxk_alpha"],
        threshold_beta=hp_config["threshold_beta"],
        warmup_steps=warmup_steps,
        decay_start=decay_start,
        steps=hp_config["training_steps"],
        normalize_to_sqrt_d=True,
    )

    # Evaluation config
    eval_cfg = NTFidelityConfig(
        eval_seq_path=eval_seq_file
        if eval_seq_file and eval_seq_file.exists()
        else None,
        model_name=model_name,
        layer_idx=hp_config["layer"],
        eval_steps=hp_config["training_steps"] + 1,  # Only evaluate at the end
        eval_batch_size=8,
        max_length=1024,
        fine_tuned_weights=str(finetuned_path) if finetuned_path else None,
    )

    # W&B config
    wandb_cfg = WandbConfig(
        use_wandb=use_wandb,
        wandb_entity=os.environ.get("WANDB_ENTITY") if use_wandb else None,
        wandb_project="sae-nt-hyperparameter-sweep" if use_wandb else None,
        wandb_name=hp_config["run_name"] if use_wandb else None,
    )

    # Checkpoint config
    checkpoint_cfg = CheckpointConfig(
        save_dir=save_dir,
        save_steps=max(
            1000, hp_config["training_steps"] // 5
        ),  # Save 5 times during training
        max_ckpts_to_keep=2,
    )

    return TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )


def create_custom_training_run(
    hp_config: Dict[str, Any],
    activation_buffer: NTActivationBuffer,
    save_base_dir: Path,
    use_wandb: bool = False,
    wandb_entity: str = None,
):
    """Create a training run that uses the activation buffer directly."""

    save_dir = save_base_dir / hp_config["run_name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer config
    warmup_steps = max(1, min(100, int(hp_config["training_steps"] * 0.1)))
    decay_start = max(warmup_steps + 1, int(hp_config["training_steps"] * 0.8))

    trainer_cfg = BatchTopKTrainerConfig(
        activation_dim=hp_config["embedding_dim"],
        dictionary_size=hp_config["dictionary_size"],
        lr=hp_config["learning_rate"],
        k=hp_config["k"],
        auxk_alpha=hp_config["auxk_alpha"],
        threshold_beta=hp_config["threshold_beta"],
        warmup_steps=warmup_steps,
        decay_start=decay_start,
        steps=hp_config["training_steps"],
        normalize_to_sqrt_d=True,
    )

    # Create trainer
    trainer = BatchTopKTrainer(trainer_cfg)

    # Create minimal managers for compatibility
    # eval_cfg = EvaluationConfig(
    #     eval_embd_dir=None, eval_steps=hp_config["training_steps"] + 1
    # )

    wandb_cfg = WandbConfig(
        use_wandb=use_wandb,
        wandb_entity=wandb_entity if use_wandb else None,
        wandb_project="sae-nt-hyperparameter-sweep" if use_wandb else None,
        wandb_name=hp_config["run_name"] if use_wandb else None,
    )

    checkpoint_cfg = CheckpointConfig(
        save_dir=save_dir,
        save_steps=max(1000, hp_config["training_steps"] // 5),
        max_ckpts_to_keep=2,
    )

    # Create a simple training run using the buffer
    class CustomTrainingRun:
        def __init__(self, trainer, activation_buffer, checkpoint_cfg, wandb_cfg):
            self.trainer = trainer
            self.data = activation_buffer
            self.checkpoint_cfg = checkpoint_cfg
            self.wandb_cfg = wandb_cfg
            self.wandb_manager = WandbManager(wandb_cfg)
            self.checkpoint_manager = CheckpointManager(checkpoint_cfg)

        def run(self):
            """Simple training loop using activation buffer."""
            if self.wandb_cfg.use_wandb:
                self.wandb_manager.init_wandb(
                    {
                        "trainer_config": self.trainer.config,
                        "activation_buffer_config": self.data.config,
                    }
                )

            print(
                f"🚀 Training {self.trainer.config.steps} steps (batch_size={self.data.out_batch_size})..."
            )

            buffer_refreshes = 0
            successful_steps = 0

            for step in tqdm(
                range(self.trainer.config.steps), desc="Training", disable=True
            ):
                try:
                    # Get batch from activation buffer
                    activations = next(self.data)
                    successful_steps += 1

                    # Training step
                    loss = self.trainer.update(step, activations)

                    # Reduced diagnostic logging every 1000 steps
                    if step % 1000 == 0:
                        print(
                            f"Step {step}/{self.trainer.config.steps}: loss={loss:.6f}"
                        )

                    # Logging
                    if self.wandb_cfg.use_wandb and step % 100 == 0:
                        self.wandb_manager.log_metrics(
                            {
                                "training/loss": loss,
                                "training/step": step,
                            },
                            step,
                        )

                    # Checkpointing
                    if step > 0 and step % self.checkpoint_cfg.save_steps == 0:
                        training_progress = {"current_step": step}
                        self.checkpoint_manager.save_checkpoint(
                            training_progress, self.trainer
                        )

                except StopIteration:
                    print(
                        f"⚠️  EARLY TERMINATION: Activation buffer exhausted at step {step}/{self.trainer.config.steps}"
                    )
                    print(f"📈 Completed {successful_steps} successful training steps")
                    print(f"🔄 Buffer refreshed {buffer_refreshes} times")
                    print(
                        f"⏱️  Training stopped at {(step / self.trainer.config.steps) * 100:.1f}% completion"
                    )
                    break

            print(f"✅ Training completed: {successful_steps} steps")
            if successful_steps < self.trainer.config.steps:
                print(
                    f"⚠️  Warning: Only completed {successful_steps}/{self.trainer.config.steps} intended steps"
                )

            # Save final model
            self.checkpoint_manager.save_final_model(self.trainer)

            if self.wandb_cfg.use_wandb:
                self.wandb_manager.finish()

    return CustomTrainingRun(trainer, activation_buffer, checkpoint_cfg, wandb_cfg)


def run_final_evaluation(
    model_path: Path, hp_config: Dict[str, Any], sequences_file: Path
) -> Dict[str, float]:
    """
    Run final evaluation computing unit-norm reconstruction loss and L0 sparsity.
    Args:
        model_path: Path to the trained SAE model (ae.pt)
        hp_config: Hyperparameter configuration used for training
        sequences_file: Sequences file for evaluation data (required)

    Returns:
        Dictionary with reconstruction_loss (unit norm MSE) and l0_sparsity metrics
    """

    try:
        print(f"🧪 Running final evaluation for {hp_config['run_name']}")
        # Load the trained model
        if not model_path.exists():
            print(f"⚠️  Model file not found: {model_path}")
            return {}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load trainer with the saved model
        trainer_config = BatchTopKTrainerConfig(
            activation_dim=hp_config["embedding_dim"],
            dictionary_size=hp_config["dictionary_size"],
            lr=hp_config["learning_rate"],
            k=hp_config["k"],
            auxk_alpha=hp_config["auxk_alpha"],
            threshold_beta=hp_config["threshold_beta"],
            steps=hp_config["training_steps"],
            warmup_steps=1,  # Set default warmup_steps
            decay_start=hp_config[
                "training_steps"
            ],  # Set decay_start to end of training
            normalize_to_sqrt_d=True,
        )

        trainer = BatchTopKTrainer(trainer_config)
        # Load the model weights directly
        trainer.ae.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        trainer.ae.eval()
        # Create evaluation data from sequences
        eval_buffer = NTActivationBuffer(
            sequences=sequences_file,
            model_name=MODEL_CONFIGS[50]["name"],  # Use NT-50M
            layer_idx=hp_config["layer"],
            n_ctxs=2000,  # Smaller sample for evaluation
            ctx_len=512,
            refresh_batch_size=32,
            out_batch_size=hp_config["batch_size"],
            device=device,
        )

        # Get evaluation batch
        eval_activations = next(iter(eval_buffer))
        eval_activations = eval_activations.to(device)
        # Run model forward pass
        with torch.no_grad():
            # Use BatchTopK (not threshold) for evaluation
            features = trainer.ae.encode(eval_activations, use_threshold=False)
            reconstructed = trainer.ae.decode(features)

        # Compute metrics
        # 1. Unit Norm Reconstruction Loss
        # Normalize both original and reconstructed to unit norm
        eval_norm = F.normalize(eval_activations, dim=-1)
        recon_norm = F.normalize(reconstructed, dim=-1)
        reconstruction_loss = F.mse_loss(recon_norm, eval_norm)

        # 2. L0 Sparsity (average non-zero features per input)
        l0_sparsity = (features != 0).float().sum(dim=-1).mean()

        # Additional metrics for context
        raw_mse = F.mse_loss(reconstructed, eval_activations)

        metrics = {
            "reconstruction_loss": reconstruction_loss.item(),  # Unit norm MSE
            "l0_sparsity": l0_sparsity.item(),  # Avg non-zero features
            "raw_mse": raw_mse.item(),  # Unnormalized MSE
            "evaluation_samples": eval_activations.shape[0],  # Sample size
        }

        # Save metrics to YAML file
        eval_file = model_path.parent / "final_evaluation.yaml"
        with open(eval_file, "w") as f:
            yaml.dump(metrics, f, indent=2, default_flow_style=False)

        print(
            f"📊 Unit Norm Recon Loss={reconstruction_loss.item():.4f}, L0={l0_sparsity.item():.1f}"
        )
        return metrics
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return {"error": str(e)}


def train_single_sae(
    hp_config: Dict[str, Any],
    model_name: str,
    save_base_dir: Path,
    eval_seq_file: Path = None,
    use_wandb: bool = False,
    wandb_entity: str = None,
    data_mode: str = "embeddings",
    sequences_file: Path = None,
    activation_batch_size: int = 32,
    finetuned_path: Path = None,
) -> Dict[str, Any]:
    """Train a single SAE with given hyperparameters"""

    start_time = datetime.now()

    try:
        print(f"Starting training: {hp_config['run_name']}")

        if data_mode == "sequences":
            # Use on-the-fly activation buffer
            print(f"Setting up activation buffer from sequences: {sequences_file}")
            activation_buffer = NTActivationBuffer(
                sequences=sequences_file,
                model_name=model_name,
                fine_tuned_weights=str(finetuned_path) if finetuned_path else None,
                layer_idx=hp_config["layer"],
                n_ctxs=30000,  # Adjust based on memory
                ctx_len=512,
                refresh_batch_size=activation_batch_size,
                out_batch_size=hp_config["batch_size"],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Create a custom training run that uses the activation buffer
            training_run = create_custom_training_run(
                hp_config=hp_config,
                activation_buffer=activation_buffer,
                save_base_dir=save_base_dir,
                use_wandb=use_wandb,
                wandb_entity=wandb_entity,
            )
        else:
            # Use pre-saved embeddings (original approach)
            config = create_training_config(
                hp_config=hp_config,
                model_name=model_name,
                save_base_dir=save_base_dir,
                eval_seq_file=eval_seq_file,
                use_wandb=use_wandb,
                data_mode=data_mode,
                sequences_file=sequences_file,
                finetuned_path=args.finetuned_path,
            )

            # Override W&B entity if provided
            if wandb_entity and use_wandb:
                config.wandb_cfg.wandb_entity = wandb_entity

            training_run = SAETrainingRun.from_config(config)

        # Run training
        training_run.run()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Run final evaluation if we have sequences and a trained model
        if data_mode == "sequences" and sequences_file:
            # Find the trained model file
            model_file = None
            if hasattr(training_run, "checkpoint_manager") and hasattr(
                training_run.checkpoint_manager, "save_dir"
            ):
                potential_model = training_run.checkpoint_manager.save_dir / "ae.pt"
                if potential_model.exists():
                    model_file = potential_model

            if model_file:
                eval_metrics = run_final_evaluation(
                    model_file, hp_config, sequences_file
                )
                print(f"✅ Final evaluation completed with {len(eval_metrics)} metrics")
            else:
                print("⚠️  No trained model found for evaluation")

        # Collect results
        # Get save directory from the training run (works for both custom and config-based runs)
        if hasattr(training_run, "checkpoint_manager") and hasattr(
            training_run.checkpoint_manager, "save_dir"
        ):
            save_dir = training_run.checkpoint_manager.save_dir
        elif data_mode != "sequences" and "config" in locals():
            save_dir = config.checkpoint_cfg.save_dir
        else:
            save_dir = save_base_dir / hp_config["run_name"]

        results = {
            "run_id": hp_config["run_id"],
            "run_name": hp_config["run_name"],
            "hyperparameters": hp_config,
            "status": "completed",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "save_dir": str(save_dir),
        }

        # Try to load final metrics
        try:
            final_eval_file = save_dir / "final_evaluation.yaml"
            if final_eval_file.exists():
                with open(final_eval_file) as f:
                    eval_results = yaml.unsafe_load(f)

                results["final_evaluation"] = eval_results

        except Exception as e:
            print(
                f"Warning: Could not load final evaluation for {hp_config['run_name']}: {e}"
            )

        print(f"Completed: {hp_config['run_name']} in {duration:.1f}s")

        # Cleanup GPU memory before next training run
        cleanup_gpu_memory()

        return results

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        error_result = {
            "run_id": hp_config["run_id"],
            "run_name": hp_config["run_name"],
            "hyperparameters": hp_config,
            "status": "failed",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "error": str(e),
        }

        print(f"FAILED: {hp_config['run_name']} after {duration:.1f}s: {e}")

        # Cleanup GPU memory even after failure to prevent accumulation
        cleanup_gpu_memory()

        return error_result


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SAE Hyperparameter Sweep for Nucleotide Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/train_sae_hyperparameter_sweep.py --model 50 --layer 6 --data-dir ./data

    # Custom hyperparameters
    python scripts/train_sae_hyperparameter_sweep.py --model 50 --layer 6 \\
        --learning-rates 1e-4 2e-4 --bandwidths 0.001 0.01 \\
        --expansion-ratios 4 8 --max-combinations 10

    # With W&B logging
    python scripts/train_sae_hyperparameter_sweep.py --model 50 --layer 6 \\
        --use-wandb --wandb-entity your-entity
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=int,
        required=True,
        choices=[50, 100, 250, 500, 2500],
        help="Model size (50, 100, 250, 500, 2500)",
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="Layer number to train SAE on"
    )

    # Data input options (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to pre-saved embeddings directory"
    )
    data_group.add_argument(
        "--sequences-file",
        type=Path,
        help="Path to DNA sequences file (FASTA, text, or CSV with 'sequence'/'sequences' column)",
    )

    # Hyperparameter customization (optional)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        help="Learning rates to sweep (default: [1e-4, 2e-4, 5e-4, 1e-3])",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        help="Number of active features per batch to sweep (default: [16, 32, 64])",
    )
    parser.add_argument(
        "--auxk-alphas",
        type=float,
        nargs="+",
        help="Auxiliary loss weights to sweep (default: [1/32])",
    )
    parser.add_argument(
        "--threshold-betas",
        type=float,
        nargs="+",
        help="Threshold update rates to sweep (default: [0.999])",
    )
    parser.add_argument(
        "--expansion-ratios",
        type=int,
        nargs="+",
        help="Expansion ratios to sweep (default: [2, 4, 8, 16])",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Batch sizes to sweep (default: [64, 128, 256, 512])",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        nargs="+",
        help="Training steps to sweep (default: [5000, 10000, 20000])",
    )

    # Other options
    parser.add_argument(
        "--max-combinations", type=int, help="Limit to random subset of combinations"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use small test configuration for debugging (100 steps, smaller batches)",
    )
    parser.add_argument(
        "--activation-batch-size",
        type=int,
        default=32,
        help="Batch size for generating activations (not SAE training batch size)",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="W&B entity name (required if using --use-wandb)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory (default: hyperparameter_sweep_results/model_layer_timestamp)",
    )
    parser.add_argument(
        "--finetuned-path",
        type=Path,
        help="Path to fine-tuned model weights (.pt file)",
    )

    args = parser.parse_args()

    # Validation
    if args.use_wandb and args.wandb_entity is None:
        parser.error("--wandb-entity is required when using --use-wandb")

    # Check file existence
    if args.data_dir and not args.data_dir.exists():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    if args.sequences_file and not args.sequences_file.exists():
        parser.error(f"Sequences file does not exist: {args.sequences_file}")

    return args


def main():
    """Main hyperparameter sweep execution"""

    args = parse_args()

    # Setup model configuration
    model_config = MODEL_CONFIGS[args.model]
    model_name = model_config["name"]
    model_short = model_config["short"]
    embedding_dim = model_config["dim"]

    # Validate embeddings directory structure if using pre-saved embeddings
    if args.data_dir:
        embeddings_dir = (
            args.data_dir / "training_embeddings" / model_short / f"layer_{args.layer}"
        )
        if not embeddings_dir.exists():
            print(
                f"❌ ERROR: Expected embeddings directory does not exist: {embeddings_dir}"
            )
            print("For embeddings mode, the data directory should contain:")
            print(
                f"  {args.data_dir}/training_embeddings/{model_short}/layer_{args.layer}/"
            )
            print(
                "Alternatively, use --sequences-file to generate embeddings on-the-fly."
            )
            return 1

    # Create custom hyperparameter grid if specified
    if args.test_mode:
        print("🧪 TEST MODE: Using small test configuration for debugging")
        base_grid = TEST_HYPERPARAMETER_GRID
    else:
        base_grid = HYPERPARAMETER_GRID

    hyperparameter_grid = {
        "learning_rates": args.learning_rates or base_grid["learning_rates"],
        "ks": args.ks or base_grid["ks"],
        "auxk_alphas": args.auxk_alphas or base_grid["auxk_alphas"],
        "threshold_betas": args.threshold_betas or base_grid["threshold_betas"],
        "expansion_ratios": args.expansion_ratios or base_grid["expansion_ratios"],
        "batch_sizes": args.batch_sizes or base_grid["batch_sizes"],
        "training_steps": args.training_steps or base_grid["training_steps"],
    }

    # Determine data mode - prefer sequences mode for robust training
    if args.sequences_file:
        data_mode = "sequences"
    elif args.data_dir and embeddings_dir.exists():
        data_mode = "embeddings"
    else:
        print("❌ ERROR: Neither sequences file nor embeddings directory available.")
        print("Please provide either:")
        print("  --sequences-file path/to/sequences.csv  (recommended)")
        print("  --data-dir path/to/embeddings_dir")
        return 1

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_source = "seq" if data_mode == "sequences" else "emb"
        output_dir = (
            Path("hyperparameter_sweep_results")
            / f"{model_short}_L{args.layer}_{data_source}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup evaluation file
    eval_seq_file = args.data_dir / "eval_sequences.txt" if args.data_dir else None

    print("=" * 80)
    print("SAE Hyperparameter Sweep")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Layer: {args.layer}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Data mode: {data_mode}")
    if data_mode == "sequences":
        print(f"Sequences file: {args.sequences_file}")
    else:
        print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"W&B logging: {args.use_wandb}")

    # Print hyperparameter grid
    print("\nHyperparameter grid:")
    for key, values in hyperparameter_grid.items():
        print(f"  {key}: {values}")
    print()

    # Generate hyperparameter combinations
    data_source = (
        args.data_dir if data_mode == "embeddings" else args.sequences_file.parent
    )
    hp_configs = generate_hyperparameter_combinations(
        model_short=model_short,
        embedding_dim=embedding_dim,
        layer=args.layer,
        data_dir=data_source,
        hyperparameter_grid=hyperparameter_grid,
        max_combinations=args.max_combinations,
    )

    print(f"Generated {len(hp_configs)} hyperparameter combinations")

    # Filter by memory constraints if using GPU
    if torch.cuda.is_available():
        gpu_memory = get_gpu_memory_gb(0)  # Check first GPU
        print(f"GPU memory: {gpu_memory:.1f}GB")
        hp_configs = filter_configs_by_memory(hp_configs, min_memory_gb=gpu_memory)
        print(f"After memory filtering: {len(hp_configs)} configurations")

    # Save hyperparameter configurations
    config_file = output_dir / "hyperparameter_configs.json"
    with open(config_file, "w") as f:
        json.dump(hp_configs, f, indent=2, default=str)
    print(f"Saved configurations to: {config_file}")
    print()

    # Run training jobs sequentially
    results = []
    print("Running training sequentially...")

    for i, hp_config in enumerate(hp_configs):
        print(f"\nProgress: {i + 1}/{len(hp_configs)}")
        print("-" * 60)

        # Monitor GPU memory before training
        if torch.cuda.is_available():
            allocated, reserved = get_current_gpu_usage()
            print(
                f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )

        result = train_single_sae(
            hp_config=hp_config,
            model_name=model_name,
            save_base_dir=output_dir / "models",
            eval_seq_file=eval_seq_file,
            use_wandb=args.use_wandb,
            wandb_entity=args.wandb_entity,
            data_mode=data_mode,
            sequences_file=args.sequences_file,
            activation_batch_size=args.activation_batch_size,
            finetuned_path=args.finetuned_path,
        )
        results.append(result)

        # Save intermediate results
        results_file = output_dir / "training_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Final results summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SWEEP COMPLETED")
    print("=" * 80)

    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"Total configurations: {len(results)}")
    print(f"Successful: {len(completed)}")
    print(f"Failed: {len(failed)}")

    if completed:
        avg_duration = sum(r["duration_seconds"] for r in completed) / len(completed)
        print(f"Average training time: {avg_duration:.1f}s")

    # Save final results
    final_results_file = output_dir / "final_results.json"
    with open(final_results_file, "w") as f:
        json.dump(
            {
                "sweep_metadata": {
                    "model": model_short,
                    "model_size": args.model,
                    "layer": args.layer,
                    "timestamp": timestamp,
                    "total_configs": len(results),
                    "successful": len(completed),
                    "failed": len(failed),
                },
                "hyperparameter_grid": hyperparameter_grid,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to: {output_dir}")
    print(f"- Configurations: {config_file}")
    print(f"- Final results: {final_results_file}")
    print(f"- Model checkpoints: {output_dir}/models/")

    if failed:
        print("\nFailed configurations:")
        for r in failed:
            print(f"  - {r['run_name']}: {r.get('error', 'Unknown error')}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
