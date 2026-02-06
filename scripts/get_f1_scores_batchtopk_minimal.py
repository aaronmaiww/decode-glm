#!/usr/bin/env python3
"""
F1 Score Analysis for BatchTopK SAE Models - Minimal Version

This script is nearly identical to the original get_f1_scores.py but uses
BatchTopK SAE architecture instead of AutoEncoder. All data loading and
validation set handling matches the original exactly.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

# Add src to path for imports (same as original)
import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

import utils
from interplm_sae.dictionary import BatchTopKSAE, IdentityDict
from measure_monosemanticity import (
    analyze_latents_fast,
    compute_metrics_across_thresholds,
    measure_monosemanticity_across_latents,
    print_metrics,
)

# Clear GPU memory (same as original)
torch.cuda.empty_cache()


def load_identity_dict(
    activation_dim: int = 512,
    device: str = "cuda",
) -> IdentityDict:
    """
    Create an IdentityDict (identity function) for analyzing raw MLP dimensions.
    """
    print(f"Creating IdentityDict for raw MLP analysis (activation_dim={activation_dim})")
    
    # Create IdentityDict instance
    sae_model = IdentityDict(activation_dim=activation_dim)
    sae_model = sae_model.to(device)
    
    print(f"✅ IdentityDict created - analyzing raw {activation_dim} MLP dimensions")
    return sae_model


def load_batchtopk_sae(
    sae_model_path: str = "path/to/batchtopk/model.pt",
    activation_dim: int = 512,
    dict_size: int = 4096,
    k: int = 32,
    device: str = "cuda",
) -> BatchTopKSAE:
    """
    Load a BatchTopK SAE model from checkpoint.
    """
    print(f"Loading BatchTopK SAE from: {sae_model_path}")
    
    # Create SAE instance
    sae_model = BatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=dict_size,
        k=k,
        normalize_to_sqrt_d=True,
    )
    
    # Load weights
    checkpoint = torch.load(sae_model_path, map_location=device, weights_only=True)
    sae_model.load_state_dict(checkpoint)
    sae_model.to(device)
    sae_model.eval()
    
    print(f"✅ Loaded BatchTopK SAE: {activation_dim}→{dict_size}, k={k}")
    return sae_model


def extract_layer_activations_and_latents_batchtopk(
    model_nt, sae_model, tokens, layer_num, batch_size=32, device="cuda"
):
    """
    Extract MLP activations and corresponding BatchTopK SAE latent representations.
    Adapted from utils.extract_layer_activations_and_latents for BatchTopK.
    """
    # Validate layer number
    max_layers = len(model_nt.esm.encoder.layer)
    if layer_num >= max_layers:
        raise ValueError(
            f"Layer number {layer_num} is out of range. Model has {max_layers} layers."
        )

    # Get model dimensions
    d_mlp = model_nt.config.hidden_size

    # Calculate batch information
    total_tokens = tokens["input_ids"].shape[0] * tokens["input_ids"].shape[1]
    num_seqs = tokens["input_ids"].shape[0]
    num_batches = (num_seqs + batch_size - 1) // batch_size

    print(f"Total tokens: {total_tokens}, num_batches: {num_batches}")

    all_latents = []
    all_acts = []
    metrics = {
        "loss": 0.0,
        "l2_loss": 0.0,
        "nmse": 0.0,
        "l1_loss": 0.0,
        "true_l0": 0.0,
        "num_samples": 0,
    }

    # Ensure models are in eval mode
    sae_model.eval()
    model_nt.eval()

    model_nt = model_nt.to(device)
    sae_model = sae_model.to(device)

    # Process batches with progress bar
    try:
        from tqdm import tqdm
        for i in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_seqs)

            # Reshape tokens for current batch
            batch_input_ids = tokens["input_ids"][start_idx:end_idx].to(device)
            batch_attention_mask = tokens["attention_mask"][start_idx:end_idx].to(device)

            with torch.no_grad():
                # Get MLP activations (same as original)
                try:
                    mlp_act = utils.get_layer_activations(
                        model_nt,
                        batch_input_ids,
                        batch_attention_mask,
                        layer_N=layer_num,
                    )

                    if mlp_act is None or len(mlp_act) == 0 or mlp_act[0].numel() == 0:
                        print(f"No or empty activations retrieved for batch {i}")
                        continue

                    mlp_act = mlp_act[0].reshape(-1, d_mlp)

                    if mlp_act.numel() == 0:
                        print(f"Empty tensor after reshaping in batch {i}")
                        continue

                    # Forward pass through SAE/IdentityDict
                    if isinstance(sae_model, BatchTopKSAE):  # BatchTopK SAE
                        latents = sae_model.encode(mlp_act, use_threshold=True, normalize_features=False)
                    else:  # IdentityDict
                        latents = sae_model.encode(mlp_act, normalize_features=False)
                    x_reconstruct = sae_model.decode(latents)
                    
                    # Compute metrics (simplified for BatchTopK)
                    l2_loss = torch.nn.functional.mse_loss(x_reconstruct, mlp_act)
                    true_l0 = (latents != 0).float().sum(dim=-1).mean()
                    
                    # Check latents before appending
                    if latents.numel() == 0:
                        print(f"Empty latents in batch {i}")
                        continue

                    all_latents.append(latents)
                    all_acts.append(mlp_act)

                    # Accumulate metrics (simplified)
                    metrics["l2_loss"] += l2_loss.item() * batch_size
                    metrics["true_l0"] += true_l0.item() * batch_size
                    metrics["num_samples"] += batch_size

                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    continue

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    finally:
        # Check if we collected any activations
        if not all_acts or not all_latents:
            raise ValueError(
                "No activations or latents were collected. Check your input data and model."
            )

        # Finalize metrics
        for key in metrics:
            if key != "num_samples":
                metrics[key] = (
                    metrics[key] / metrics["num_samples"]
                    if metrics["num_samples"] > 0
                    else 0
                )

        # Combine results, move to cpu before
        all_acts = torch.cat(all_acts, dim=0).cpu()
        all_latents = [x.cpu() for x in all_latents]
        combined_latents = torch.cat(all_latents, dim=0).cpu()

        # Clean up
        torch.cuda.empty_cache()

    return all_acts, combined_latents, metrics


def run_latent_analysis(
    sae_model: Any,
    model_nt: Any,
    tokens: List,
    token_df: pd.DataFrame,
    val_set_idx: int,
    previous_results_df: Optional[pd.DataFrame] = None,
    layer_num: int = 11,
    batch_size: int = 32,
    f1_type: str = "domain",
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Run the full latent analysis pipeline for a given validation set.
    This is IDENTICAL to the original except for the activation extraction function.
    """

    # Extract layer activations and SAE latents (ONLY DIFFERENCE: use BatchTopK function)
    all_acts, combined_latents, metrics = extract_layer_activations_and_latents_batchtopk(
        model_nt=model_nt,
        sae_model=sae_model,
        tokens=tokens,
        layer_num=layer_num,
        batch_size=batch_size,
        device=device,
    )

    optimal_thresholds_dict = None

    # For first validation set, analyze all latents to identify candidates
    if previous_results_df is None:
        print("Running latent discovery for validation set 0...")
        latent_dict = analyze_latents_fast(
            combined_latents=combined_latents,
            token_df=token_df,
            sae_model=sae_model,
            top_n=20,
            min_threshold=10,
            batch_size=2000,
        )

        # Ensure that values in latent_dict are each a list of strings
        for k, v in latent_dict.items():
            if isinstance(v, set):
                latent_dict[k] = list(v)
            elif not isinstance(v, list):
                latent_dict[k] = [str(v)]

    # For subsequent validation sets, use candidates and activation thresholds
    # from the first validation set
    else:
        print(
            f"Using results from previous validation set for validation set {val_set_idx}..."
        )

        # Extract latent IDs and annotations from previous results
        if (
            "latent_id" in previous_results_df.columns
            and "annotation" in previous_results_df.columns
        ):
            latent_dict = {}
            for _, row in previous_results_df.iterrows():
                latent_id = row["latent_id"]
                annotation = row["annotation"]
                # Handle different annotation formats
                if isinstance(annotation, list):
                    latent_dict[latent_id] = annotation
                elif isinstance(annotation, str):
                    latent_dict[latent_id] = [annotation]
                else:
                    latent_dict[latent_id] = [str(annotation)]

            # Extract optimal thresholds
            if "threshold" in previous_results_df.columns:
                optimal_thresholds_dict = {}
                for _, row in previous_results_df.iterrows():
                    latent_id = row["latent_id"]
                    threshold = row["threshold"]
                    optimal_thresholds_dict[latent_id] = [threshold]
        else:
            raise ValueError("Previous results DataFrame missing required columns")

        print(f"Using {len(latent_dict)} latents from previous results")
        if optimal_thresholds_dict:
            print(
                f"Using {len(optimal_thresholds_dict)} optimal thresholds from previous results"
            )

    # Measure monosemanticity for each latent-annotation pair
    print(
        f"Calculating F1-scores for validation set {val_set_idx} using {f1_type} method..."
    )

    # Convert f1_type to boolean for existing function
    use_modified_recall = f1_type == "domain"

    # Create wrapper function to pass the modified_recall parameter
    def compute_metrics_with_f1_type(
        token_df, annotation, latent_id, thresholds=None, modified_recall=True
    ):
        return compute_metrics_across_thresholds(
            token_df=token_df,
            annotation=annotation,
            latent_id=latent_id,
            thresholds=thresholds,
            modified_recall=use_modified_recall,
        )

    results_df = measure_monosemanticity_across_latents(
        latent_dict=latent_dict,
        combined_latents=combined_latents,
        token_df=token_df,
        compute_metrics_across_thresholds=compute_metrics_with_f1_type,
        print_metrics=print_metrics,
        validation_set=val_set_idx,
        optimal_thresholds_dict=optimal_thresholds_dict,
    )

    print(
        f"Completed analysis for validation set {val_set_idx}. Found {len(results_df)} results."
    )
    return results_df


def save_results(
    results_df: pd.DataFrame,
    validation_set: int,
    output_dir: Path,
    model_type: str,
    f1_type: str,
) -> str:
    """Save validation set results to CSV file. (IDENTICAL to original)"""
    filename = f"val_set_{validation_set}_{model_type}_{f1_type}_results.csv"
    filepath = output_dir / filename
    results_df.to_csv(filepath, index=False)
    print(f"Saved validation set {validation_set} results to: {filepath}")
    return str(filepath)


def save_summary(
    all_results: Dict[str, pd.DataFrame],
    output_dir: Path,
    model_type: str,
    f1_type: str,
    run_metadata: Dict[str, Any],
) -> str:
    """Save summary statistics and metadata. (IDENTICAL to original)"""
    # Calculate summary statistics
    summary_stats = {}
    for val_set, df in all_results.items():
        if not df.empty:
            f1_col = f"best_f1_{val_set.replace('val_', 'val')}"
            summary_stats[val_set] = {
                "mean_f1": float(df[f1_col].mean()),
                "max_f1": float(df[f1_col].max()),
                "min_f1": float(df[f1_col].min()),
                "std_f1": float(df[f1_col].std()),
                "num_latents": len(df),
                "top_f1_latents": df.nlargest(5, f1_col)[["latent_id", f1_col]].to_dict(
                    "records"
                ),
            }

    # Combine metadata and summary
    summary_data = {
        "run_metadata": run_metadata,
        "summary_statistics": summary_stats,
    }

    # Save summary as JSON
    summary_filepath = output_dir / f"summary_{model_type}_{f1_type}.json"
    with open(summary_filepath, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Saved summary to: {summary_filepath}")
    return str(summary_filepath)


def load_models_and_data(
    model_type: str = "batchtopk", 
    device: str = "cuda",
    sae_model_path: str = None,
    activation_dim: int = 512,
    dict_size: int = 4096,
    k: int = 32,
    model_name: str = "nt50",
) -> Dict[str, Any]:
    """
    Load models and data. Similar to original but loads BatchTopK SAE.
    """
    try:
        # Load the model and tokenizer with specified model size
        model_size = model_name.replace("nt", "") + "m" if model_name.startswith("nt") else "50m"
        model_nt, tokenizer = utils.load_model_and_tokenizer(model_size=model_size)

        # Load the SAE model 
        if model_type == "batchtopk":
            if sae_model_path is None:
                raise ValueError("sae_model_path is required for BatchTopK SAE")
            sae_model = load_batchtopk_sae(
                sae_model_path=sae_model_path,
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                device=device
            )
        elif model_type == "identity":
            sae_model = load_identity_dict(
                activation_dim=activation_dim,
                device=device
            )
        else:
            raise ValueError("This script supports 'batchtopk' and 'identity' model types")

        # Load token data for validation sets (IDENTICAL to original)
        tokens_s0, token_df_1k_s0 = utils.load_validation_set(0)
        tokens_s1, token_df_1k_s1 = utils.load_validation_set(1)
        tokens_s2, token_df_1k_s2 = utils.load_validation_set(2)

        return {
            "model_nt": model_nt,
            "tokenizer": tokenizer,
            "sae_model": sae_model,
            "validation_sets": [
                (tokens_s0, token_df_1k_s0),
                (tokens_s1, token_df_1k_s1),
                (tokens_s2, token_df_1k_s2),
            ],
        }
    except ImportError as e:
        print(f"Error loading models/data: {e}")
        print("Please ensure utils.py is available and all dependencies are installed")
        raise


def parse_arguments():
    """Parse command line arguments. (Similar to original with BatchTopK params)"""
    parser = argparse.ArgumentParser(
        description="Run F1-score analysis on BatchTopK SAE latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--f1-type",
        choices=["domain", "standard"],
        default="domain",
        help="F1 calculation method: 'domain' for domain-based (modified recall), 'standard' for token-level",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for processing (reduce if encountering CUDA OOM errors)",
    )

    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )

    parser.add_argument(
        "--layer-num",
        type=int,
        default=11,
        help="Layer number to extract activations from",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the analysis on (e.g., 'cuda' or 'cpu')",
    )

    # BatchTopK SAE specific arguments
    parser.add_argument(
        "--model-type",
        choices=["batchtopk", "identity"],
        default="batchtopk",
        help="Model type: 'batchtopk' for BatchTopK SAE, 'identity' for raw MLP analysis",
    )
    
    parser.add_argument(
        "--sae-model-path",
        type=str,
        required=False,
        help="Path to BatchTopK SAE model weights file (.pt) - required for batchtopk, ignored for identity",
    )
    
    parser.add_argument(
        "--activation-dim",
        type=int,
        default=512,
        help="Activation dimension (input size to SAE)",
    )
    
    parser.add_argument(
        "--dict-size",
        type=int,
        default=4096,
        help="Dictionary size (number of SAE features)",
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Top-k parameter for BatchTopK SAE",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="nt50",
        help="Name of the model being analyzed (e.g., nt50, nt100)",
    )

    return parser.parse_args()


def create_output_directory(base_dir: str, model_type: str, f1_type: str, model_name: str, layer_num: int) -> Path:
    """Create timestamped output directory with model and layer info."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{timestamp}_{model_name}_L{layer_num}_{model_type}_{f1_type}"
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    return output_dir


def main():
    """
    Main function to run the complete F1-score analysis pipeline.
    This is IDENTICAL to the original except for model loading.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    if args.model_type == "batchtopk" and args.sae_model_path is None:
        raise ValueError("--sae-model-path is required when using --model-type batchtopk")

    # Create output directory with model and layer info
    output_dir = create_output_directory(args.output_dir, args.model_type, args.f1_type, args.model_name, args.layer_num)

    # Save run metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": args.model_type,
        "f1_type": args.f1_type,
        "batch_size": args.batch_size,
        "layer_num": args.layer_num,
        "sae_model_path": args.sae_model_path,
        "activation_dim": args.activation_dim,
        "dict_size": args.dict_size,
        "k": args.k,
        "model_name": args.model_name,
        "command_line_args": vars(args),
    }

    try:
        # Load models and data (DIFFERENT: passes BatchTopK params)
        print("Loading models and data...")
        data = load_models_and_data(
            model_type=args.model_type, 
            device=args.device,
            sae_model_path=args.sae_model_path,
            activation_dim=args.activation_dim,
            dict_size=args.dict_size,
            k=args.k,
            model_name=args.model_name,
        )

        model_nt = data["model_nt"]
        sae_model = data["sae_model"]
        validation_sets = data["validation_sets"]

        results = {}
        previous_results = None
        saved_files = []

        # Run analysis on each validation set (IDENTICAL to original)
        for val_set in range(3):
            tokens, token_df = validation_sets[val_set]

            print(f"\n{'=' * 50}")
            print(f"ANALYZING VALIDATION SET {val_set}")
            print(f"{'=' * 50}")

            results_df = run_latent_analysis(
                sae_model=sae_model,
                model_nt=model_nt,
                tokens=tokens,
                token_df=token_df,
                val_set_idx=val_set,
                previous_results_df=previous_results,
                layer_num=args.layer_num,
                batch_size=args.batch_size,
                f1_type=args.f1_type,
                device=args.device,
            )

            results[f"val_{val_set}"] = results_df
            previous_results = results_df

            # Save results immediately
            saved_file = save_results(
                results_df, val_set, output_dir, f"{args.model_name}_{args.model_type}", args.f1_type
            )
            saved_files.append(saved_file)

            # Print summary
            if not results_df.empty:
                f1_col = f"best_f1_val{val_set}"
                mean_f1 = results_df[f1_col].mean()
                max_f1 = results_df[f1_col].max()
                print(f"Validation Set {val_set} Summary:")
                print(f"  - Mean F1: {mean_f1:.3f}")
                print(f"  - Max F1: {max_f1:.3f}")
                print(f"  - Latents analyzed: {len(results_df)}")

        # Save summary statistics
        summary_file = save_summary(
            results, output_dir, f"{args.model_name}_{args.model_type}", args.f1_type, run_metadata
        )
        saved_files.append(summary_file)

        # Print final summary (IDENTICAL to original)
        print(f"\n{'=' * 60}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 60}")
        print(f"Model type: {args.model_type}")
        print(f"F1 calculation: {args.f1_type}")
        print(f"Results saved to: {output_dir}")
        print(f"Files created: {len(saved_files)}")
        for file in saved_files:
            print(f"  - {file}")

        return results

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    if not results:
        print("Analysis failed. Please check error messages above.")
        exit(1)