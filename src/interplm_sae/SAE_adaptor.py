"""this is a wrapper function that adpats the SAE model we use so that it has certain methods that are needed for the steering code
And so that the method name matches the one used in the steering code
"""
import torch
from typing import Tuple

class SAEWrapper:
    """
    Wrapper to provide consistent interface for both AutoEncoder and BatchTopKSAE.
    Handles differences in architecture, forward pass outputs, and decoding.
    """
    
    def __init__(self, sae, device: str = "cuda"):
        """
        Initialize wrapper and auto-detect SAE type.
        
        Args:
            sae: Either an AutoEncoder or BatchTopKSAE instance
            device: Device to use for operations
        """
        self.sae = sae
        self.device = device
        
        # Auto-detect SAE type by checking for distinctive attributes
        if hasattr(sae, 'W_enc') and hasattr(sae, 'W_dec'):
            self.sae_type = "AutoEncoder"
            print(f"[SAEWrapper] Initialized with AutoEncoder")
        elif hasattr(sae, 'encoder') and hasattr(sae, 'decoder'):
            self.sae_type = "BatchTopKSAE"
            print(f"[SAEWrapper] Initialized with BatchTopKSAE (k={sae.k.item()})")
        else:
            raise ValueError(f"Unknown SAE type. Available attributes: {dir(sae)}")
    
    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both reconstruction and latents in a single forward pass.
        
        Args:
            x: Input activations [seq_len, d_model] or [batch, seq_len, d_model]
            
        Returns:
            Tuple of (reconstruction, latents):
                - reconstruction: Reconstructed activations (same shape as input)
                - latents: Encoded latent activations [seq_len, dict_size] or [batch, seq_len, dict_size]
        """
        if self.sae_type == "AutoEncoder":
            # AutoEncoder returns: (loss, x_reconstruct, acts_sparse, l2_loss, nmse, l1_loss, true_l0)
            _, recon, latents, _, _, _, _ = self.sae(x)
            return recon, latents
        else:  # BatchTopKSAE
            # BatchTopKSAE with output_features=True returns: (x_hat, encoded_acts)
            recon, latents = self.sae(x, output_features=True)
            return recon, latents
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to latent space.
        
        Args:
            x: Input activations
            
        Returns:
            Encoded latent activations
        """
        if self.sae_type == "AutoEncoder":
            # Forward pass and extract latents (3rd output)
            _, _, latents, _, _, _, _ = self.sae(x)
            return latents
        else:  # BatchTopKSAE
            # Direct encode method
            return self.sae.encode(x)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to activation space.
        
        Args:
            latents: Latent activations to decode
            
        Returns:
            Decoded activations
        """
        if self.sae_type == "AutoEncoder":
            # Direct matrix multiplication: latents @ W_dec + b_dec
            return latents @ self.sae.W_dec + self.sae.b_dec
        else:  # BatchTopKSAE
            # Use decode method (internally: decoder(latents) + b_dec)
            # decoder is nn.Linear, so this is: latents @ decoder.weight.T + b_dec
            return self.sae.decode(latents)
    
    def get_decoder_weights(self) -> torch.Tensor:
        """
        Get decoder weight matrix in consistent format [hidden_dim, activation_dim].
        
        Returns:
            Decoder weight matrix
        """
        if self.sae_type == "AutoEncoder":
            return self.sae.W_dec
        else:  # BatchTopKSAE
            # nn.Linear stores weights as [out, in], need to transpose
            return self.sae.decoder.weight.T
    
    def get_decoder_bias(self) -> torch.Tensor:
        """
        Get decoder bias vector.
        
        Returns:
            Decoder bias
        """
        return self.sae.b_dec