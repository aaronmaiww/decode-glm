"""
Calculate the loss (cross entropy) fidelity metric for a Sparse Autoencoder (SAE) trained on Nucleotide Transformer embeddings.
Adapted from the ESM fidelity implementation for genomic sequences.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from nnsight import NNsight
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from interplm_sae.nt_intervention import get_nt_output_with_intervention
from .evaluation import EvaluationConfig, EvaluationManager


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class NTFidelityConfig(EvaluationConfig):
    """Fidelity evaluation config for Nucleotide Transformer models."""

    model_name: str | None = None
    layer_idx: int | None = None
    max_length: int = 1024
    fine_tuned_weights: str | None = None

    def build(self) -> "NTFidelityFunction":
        return NTFidelityFunction(self)


def calculate_cross_entropy(model_output, batch_tokens, batch_attn_mask):
    """Calculate cross entropy for each sequence in batch, excluding start/end tokens."""
    losses = []
    for j, mask in enumerate(batch_attn_mask):
        length = mask.sum()
        seq_logits = model_output[j, 1 : length - 1]
        seq_tokens = batch_tokens[j, 1 : length - 1]
        loss = F.cross_entropy(seq_logits, seq_tokens)
        losses.append(loss.item())
    return losses


def calculate_loss_recovered(ce_autoencoder, ce_identity, ce_zero_ablation):
    """
    Calculate the loss recovered metric for a Sparse Autoencoder (SAE).

    If the recovered loss is as good as the original loss as possible, the
    metric will be 100%. If the recovered loss is as bad as zero-ablating, the
    metric will be 0%.

    Parameters:
    ce_autoencoder (float): Cross-entropy loss when using the SAE's reconstructions
    ce_identity (float): Cross-entropy loss when using the identity function
    ce_zero_ablation (float): Cross-entropy loss when using the zero-ablation function

    Returns:
    float: The loss recovered metric as a percentage
    """
    numerator = ce_autoencoder - ce_identity
    denominator = ce_zero_ablation - ce_identity

    # Avoid division by zero
    if np.isclose(denominator, 0):
        return 0.0

    loss_recovered = 1 - (numerator / denominator)

    # Clip the result to be between 0 and 1
    loss_recovered = np.clip(loss_recovered, 0, 1)

    # Convert to percentage
    return loss_recovered * 100


class NTFidelityFunction(EvaluationManager):
    """Fidelity evaluation for Nucleotide Transformer models."""

    def __init__(
        self,
        eval_config: "NTFidelityConfig",
    ):
        # The super class just sets the config
        super().__init__(eval_config)

        print("Prepping Nucleotide Transformer loss fidelity function")
        self.device = get_device()

        # Extract config values
        self.model_name = eval_config.model_name
        self.eval_seq_path = eval_config.eval_seq_path
        self.layer_idx = eval_config.layer_idx
        self.batch_size = eval_config.eval_batch_size or 8
        self.max_length = eval_config.max_length
        self.fine_tuned_weights = eval_config.fine_tuned_weights

        # Load the Nucleotide Transformer model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load fine-tuned weights if provided
        if self.fine_tuned_weights:
            print(f"Loading fine-tuned weights from: {self.fine_tuned_weights}")
            checkpoint = torch.load(self.fine_tuned_weights, map_location=self.device)
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Load evaluation sequences
        with open(self.eval_seq_path) as f:
            eval_seqs = [line.strip() for line in f]

        # Pre-tokenize and create batches
        self.tokenized_batches = []
        for i in range(0, len(eval_seqs), self.batch_size):
            batch_seqs = eval_seqs[i : i + self.batch_size]

            # Clean sequences (remove invalid nucleotides)
            batch_seqs = [self._preprocess_sequence(seq) for seq in batch_seqs]

            # Tokenize
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            batch_tokens = inputs["input_ids"]
            batch_mask = inputs["attention_mask"]
            self.tokenized_batches.append((batch_tokens, batch_mask))

        self.nnsight_model = NNsight(self.model, device=self.device)

        self.orig_loss, self.zero_loss = self._CE_for_orig_and_zero_ablation(
            self.tokenized_batches
        )
        print("Finished initializing NT Fidelity Function")

    def _preprocess_sequence(self, sequence: str) -> str:
        """Clean and validate DNA sequence."""
        import re

        # Remove whitespace and make uppercase
        sequence = sequence.strip().upper()
        # Remove invalid nucleotides (keep only A, T, G, C, N)
        sequence = re.sub(r"[^ATGCN]", "N", sequence)
        return sequence

    def _calculate_fidelity(self, sae_model) -> dict:
        """Calculate fidelity metrics for the SAE model."""
        sae_loss = self._CE_from_sae_recon(self.tokenized_batches, sae_model)

        loss_recovered = calculate_loss_recovered(
            ce_autoencoder=sae_loss,
            ce_identity=self.orig_loss,
            ce_zero_ablation=self.zero_loss,
        )

        return {"pct_loss_recovered": loss_recovered, "CE_w_sae_patching": sae_loss}

    def _CE_for_orig_and_zero_ablation(self, tokenized_batches):
        """Calculate cross entropy for original and zero-ablated outputs."""
        orig_losses, zero_losses = [], []

        for batch_tokens, batch_attn_mask in tqdm(tokenized_batches):
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            orig_logits, orig_hidden = get_nt_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
            )

            zero_logits, _ = get_nt_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
                torch.zeros_like(orig_hidden),
            )

            orig_losses.extend(
                calculate_cross_entropy(orig_logits, batch_tokens, batch_attn_mask)
            )
            zero_losses.extend(
                calculate_cross_entropy(zero_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(orig_losses), np.mean(zero_losses)

    def _CE_from_sae_recon(self, tokenized_batches, sae_model):
        """Calculate cross entropy using SAE reconstructions."""
        sae_losses = []

        for batch_tokens, batch_attn_mask in tokenized_batches:
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            _, orig_hidden = get_nt_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
            )

            # Get reconstructions with unnormalize=True for injection into the model
            reconstructions = sae_model(orig_hidden, unnormalize=True)
            sae_logits, _ = get_nt_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
                reconstructions,
            )

            sae_losses.extend(
                calculate_cross_entropy(sae_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(sae_losses)
