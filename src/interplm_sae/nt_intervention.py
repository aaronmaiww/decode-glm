"""
Nucleotide Transformer-specific intervention functions.
Adapted from ESM intervention for genomic language models.
"""

import torch
from nnsight import NNsight
from transformers import AutoModelForMaskedLM


def get_nt_submodule_and_access_method(nnsight_model: NNsight, hidden_layer_idx: int):
    """Get the submodule at the given hidden layer index and the access method (input or output)

    For Nucleotide Transformer models, we follow a similar approach to ESM-2 where we
    access the input of the next layer to effectively patch the output of the current layer.
    """

    # Access the transformer layers - NT models use a different structure than ESM
    # NT models have: nucleotide_transformer.encoder.layer[i]
    n_layers = len(nnsight_model.nucleotide_transformer.encoder.layer)

    # Confirm the hidden layer index is within bounds
    if hidden_layer_idx > n_layers:
        raise ValueError(f"Hidden layer index {hidden_layer_idx} is out of bounds")

    # The last hidden layer may have additional normalization
    elif hidden_layer_idx == n_layers:
        # For NT models, check if there's a final layer norm
        if hasattr(nnsight_model.nucleotide_transformer.encoder, "layer_norm"):
            return nnsight_model.nucleotide_transformer.encoder.layer_norm, "output"
        else:
            # Fallback to the last layer's output
            return nnsight_model.nucleotide_transformer.encoder.layer[
                hidden_layer_idx - 1
            ], "output"

    # For intermediate layers, access the input of the next layer
    else:
        return nnsight_model.nucleotide_transformer.encoder.layer[
            hidden_layer_idx
        ], "input"


def get_nt_output_with_intervention(
    nt_model: AutoModelForMaskedLM,
    nnsight_model: NNsight,
    batch_tokens: torch.Tensor,
    batch_attn_mask: torch.Tensor,
    hidden_layer_idx: int,
    hidden_state_override: torch.Tensor | None = None,
):
    """Get Nucleotide Transformer model output with optional hidden state modification."""
    with torch.no_grad():
        orig_output = nt_model(
            batch_tokens, attention_mask=batch_attn_mask, output_hidden_states=True
        )

        if hidden_state_override is None:
            return orig_output.logits, orig_output.hidden_states[hidden_layer_idx]

        submodule, input_or_output = get_nt_submodule_and_access_method(
            nnsight_model, hidden_layer_idx
        )

        with nnsight_model.trace(
            batch_tokens, attention_mask=batch_attn_mask
        ) as tracer:
            embd_to_patch = (
                submodule.input[0][0]
                if input_or_output == "input"
                else submodule.output
            )
            embd_to_patch[:] = hidden_state_override
            modified_logits = nnsight_model.output.save()

        return modified_logits.value.logits, orig_output.hidden_states[hidden_layer_idx]
