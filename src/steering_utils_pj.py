from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.interplm_sae.dictionary import BatchTopKSAE
from src.interplm_sae.SAE_adaptor import SAEWrapper


def load_sae_universal(path: str, device: str = "cuda"):
    """
    Universal SAE loader that handles both AutoEncoder and BatchTopKSAE.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    
    # Detect model type by checking keys
    if "encoder.weight" in checkpoint or (
        "model_state_dict" in checkpoint 
        and "encoder.weight" in checkpoint["model_state_dict"]
    ):
        print("Detected BatchTopKSAE format")
        # Handle BatchTopKSAE
        return BatchTopKSAE.from_pretrained(path, device=device)
    
    elif "W_enc" in checkpoint or (
        "model_state_dict" in checkpoint 
        and "W_enc" in checkpoint["model_state_dict"]
    ):
        print("Detected AutoEncoder format")
        # Load using your current method
        from utils import load_sae_model
        return load_sae_model(path, device)
    
    else:
        raise ValueError(f"Unknown SAE format. Keys: {list(checkpoint.keys())}")

def steer_with_sae(
    sae_model_path: str,
    latents_to_max: List[int],
    latents_to_zero: List[int],
    input_sequence: str = "TAAA" * 10,
    layer_num: int = 11,
    device: str = "cpu",
    top_k: int = 15,
    steering_value: int = 1,
    steering_value_method: str = "fixed",#alternative is "max_activation"
    model_name: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    position_to_steer: int = None,
) -> Dict[str, any]:
    """
    Steer model by manipulating SAE latents and return logit changes.
    
    Args:
        sae_model_path: Path to SAE weights
        latents_to_max: Latent indices to maximize
        latents_to_zero: Latent indices to zero out
        input_sequence: Input sequence to steer
        layer_num: Layer to modify
        device: 'cpu' or 'cuda'
        top_k: Number of top changed logits to return
        model_name: Name of the model to use for steering
        steering_value: Value to steer the latents to
        steering_value_method: Method to use to determine the steering value
    Returns:
        Dict with not steered logits, steered logits, top_increased_logits, top_decreased_logits, and mean_logit_diff (mean over sequence length)
    """
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device).eval()
    
    
    # Load SAE - the cfg is hard codded for now as was using the SAE's provided - might need to make this a argument to make it compatible with the new SAE's
    sae = load_sae_universal(sae_model_path, device)
    #define the sae using the SAEWrapper so that we have compatible methods
    sae = SAEWrapper(load_sae_universal(sae_model_path, device))
    
    # Capture activations this will be used as input to the SAE and then to get the residual of the reconstruction
    captured = {}
    def capture_hook(name):
        def hook(module, input, output):
            captured[name] = output.clone()
        return hook
    
    hook = model.esm.encoder.layer[layer_num].output.dense.register_forward_hook(
        capture_hook("acts")
    )
    
    input_ids = tokenizer(input_sequence, return_tensors="pt")["input_ids"].to(device)
    
    #non-steered forward pass to get the activations (using hook) and also the logits of a non-steered forward pass
    with torch.no_grad():
        logits_control = model(input_ids)['logits'].detach().cpu().numpy()
    
    hook.remove()
    acts = captured["acts"].squeeze(0) if captured["acts"].dim() == 3 else captured["acts"]
    
    # Pass through SAE and manipulate latents
    #obtain the latents and the reconstruction
    with torch.no_grad():
        recon, latents = sae.reconstruct(acts)# we will use both the latents for the steering and the non-steered reconstruction to get the residual which we will add back to the steered reconstruction
    
    residual = acts - recon
    clamped_latents = latents.clone()
    if steering_value_method == "max_activation":
        max_val = latents.max()
    else:
        max_val = steering_value
    
    # manipulate the latents
    if position_to_steer is not None:
        for position in position_to_steer:
            for idx in latents_to_zero:
                clamped_latents[position, idx] = 0
            for idx in latents_to_max:
                clamped_latents[position, idx] = max_val
    else:
        for idx in latents_to_zero:
            clamped_latents[:, idx] = 0
        for idx in latents_to_max:
            clamped_latents[:, idx] = max_val
    

    steered_acts = sae.decode(clamped_latents) + residual
    
    # Substitute activations and run forward pass
    def substitute_hook(sub_acts):
        def hook(module, input, output):
            return sub_acts
        return hook
    
    batched_acts = steered_acts.unsqueeze(0) if steered_acts.dim() == 2 else steered_acts
    model.esm.encoder.layer[layer_num].output.dense._forward_hooks.clear()
    
    hook = model.esm.encoder.layer[layer_num].output.dense.register_forward_hook(
        substitute_hook(batched_acts)
    )
    
    # forward pass with the steered activations
    with torch.no_grad():
        logits_steered = model(input_ids)['logits'].detach().cpu().numpy()
    
    hook.remove()
    model.esm.encoder.layer[layer_num].output.dense._forward_hooks.clear()
    
    
    # Compute differences
    logits_steered = logits_steered.squeeze(0) if logits_steered.ndim == 3 else logits_steered
    logits_control = logits_control.squeeze(0) if logits_control.ndim == 3 else logits_control
    diff = logits_steered - logits_control
    mean_diff = diff.mean(axis=0)
    
    # Get top changed tokens
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(mean_diff))))
    
    top_inc_idx = np.argpartition(mean_diff, -top_k)[-top_k:]
    top_increased = sorted(
        [(mean_diff[i], int(i), vocab[i]) for i in top_inc_idx],
        reverse=True
    )
    
    top_dec_idx = np.argpartition(mean_diff, top_k)[:top_k]
    top_decreased = sorted(
        [(mean_diff[i], int(i), vocab[i]) for i in top_dec_idx]
    )
    
    return {
        "logits_steered": logits_steered,
        "logits_control": logits_control,
        'top_increased_logits': top_increased,
        'top_decreased_logits': top_decreased,
        'mean_logit_diff': mean_diff,
        'max_value': max_val,
    }

