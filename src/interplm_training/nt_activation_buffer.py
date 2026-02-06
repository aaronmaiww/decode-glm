"""
Nucleotide Transformer Activation Buffer for on-the-fly embedding generation.

Adapted from dictionary_learning/buffer.py to work with NT models and DNA sequences.
"""

import gc
import re
from pathlib import Path
from typing import Iterator, List, Optional, Union

import pandas as pd
import torch
import torch as t
from nnsight import LanguageModel
from transformers import AutoModelForMaskedLM, AutoTokenizer


class NTActivationBuffer:
    """
    Activation buffer for Nucleotide Transformer models that generates embeddings on-the-fly
    from DNA sequences. Handles DNA-specific preprocessing and tokenization.

    This is adapted from the dictionary_learning ActivationBuffer to work specifically
    with Nucleotide Transformer models and genomic sequences.
    """

    def __init__(
        self,
        sequences: Union[
            str, Path, List[str], Iterator[str]
        ],  # sequences file or list/iterator
        model_name: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        fine_tuned_weights: Optional[str] = None,  # path to fine-tuned weights
        layer_idx: int = 6,  # which layer to extract from
        n_ctxs: int = 30000,  # approximate number of contexts to store
        ctx_len: int = 512,  # length of each sequence context
        refresh_batch_size: int = 32,  # batch size for model forward passes
        out_batch_size: int = 8192,  # batch size for yielding activations
        device: str = "cuda",  # device for storing activations
        max_length: int = 1024,  # max sequence length for tokenizer
        remove_special_tokens: bool = True,  # remove CLS/EOS tokens
    ):
        self.model_name = model_name
        self.fine_tuned_weights = fine_tuned_weights
        self.layer_idx = layer_idx
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.max_length = max_length
        self.remove_special_tokens = remove_special_tokens

        # Load model and tokenizer
        print(f"Loading Nucleotide Transformer model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
            else:
                # Fallback: use a special pad token
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                print(f"Added new pad_token: {self.tokenizer.pad_token}")
        else:
            print(f"Tokenizer already has pad_token: {self.tokenizer.pad_token}")

        # Double check pad_token_id
        print(f"Pad token ID: {self.tokenizer.pad_token_id}")

        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

        # Load fine-tuned weights if provided
        if fine_tuned_weights:
            print(f"Loading fine-tuned weights from: {fine_tuned_weights}")
            checkpoint = torch.load(
                fine_tuned_weights, map_location="cpu", weights_only=True
            )

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                fine_tuned_state_dict = checkpoint["model_state_dict"]
                print(
                    f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
                )
            elif isinstance(checkpoint, dict) and any(
                key.startswith(("esm.", "encoder.", "embeddings."))
                for key in checkpoint
            ):
                # Direct state dict format
                fine_tuned_state_dict = checkpoint
            else:
                raise ValueError(
                    f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())}"
                )

            model.load_state_dict(fine_tuned_state_dict, strict=False)
            print("✅ Fine-tuned weights loaded successfully")

        # Explicitly move model to device before wrapping with LanguageModel
        model = model.to(device)
        self.model = LanguageModel(model, device_map=device)

        # Verify model is on correct device
        model_device = next(self.model.parameters()).device
        print(f"Model moved to device: {model_device}")
        if str(model_device) != str(device):
            print(f"WARNING: Model device {model_device} != target device {device}")
        else:
            print(f"✅ Model successfully on target device: {device}")

        # Get the target layer for activation extraction
        self.target_layer = self.model.esm.encoder.layer[layer_idx]

        # Determine activation dimension
        self.d_model = self._get_model_dim()

        # Initialize activation buffer
        self.activation_buffer_size = n_ctxs * ctx_len
        self.activations = t.empty(
            0, self.d_model, device=device, dtype=self.model.dtype
        )
        self.read = t.zeros(0).bool()

        # Setup sequence iterator
        self.sequence_iterator = self._setup_sequence_iterator(sequences)

        print("NT Activation Buffer initialized:")
        print(f"  Model: {model_name}")
        print(f"  Layer: {layer_idx}")
        print(f"  Activation dim: {self.d_model}")
        print(f"  Buffer size: {self.activation_buffer_size:,} activations")
        print(f"  Context length: {ctx_len}")

    def _get_model_dim(self) -> int:
        """Get the model's hidden dimension."""
        model_dims = {
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species": 512,
            "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species": 512,
            "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species": 768,
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species": 1024,
            "InstaDeepAI/nucleotide-transformer-v2-2.5b-multi-species": 2560,
        }
        return model_dims.get(self.model_name, 512)  # default to 512 if unknown

    def _setup_sequence_iterator(self, sequences) -> Iterator[str]:
        """Setup iterator for DNA sequences from various input types."""
        if isinstance(sequences, (str, Path)):
            # File path - load sequences from file
            sequences_file = Path(sequences)
            if not sequences_file.exists():
                raise FileNotFoundError(f"Sequences file not found: {sequences_file}")

            # Handle different file formats
            if sequences_file.suffix.lower() == ".csv":
                return self._csv_iterator(sequences_file)
            else:
                return self._text_or_fasta_iterator(sequences_file)
        elif isinstance(sequences, list):
            # List of sequences - cycle through them indefinitely
            print(f"Found {len(sequences)} sequences in list")
            import itertools

            return itertools.cycle(sequences)

        elif hasattr(sequences, "__iter__"):
            # Already an iterator
            return sequences

        else:
            raise ValueError(
                "sequences must be a file path, list of sequences, or iterator"
            )

    def _csv_iterator(self, csv_path: Path) -> Iterator[str]:
        """Iterator for CSV files with 'sequence' or 'sequences' column."""
        df = pd.read_csv(csv_path)

        # Look for sequence column (case insensitive)
        sequence_col = None
        for col in df.columns:
            if col.lower() in ["sequence", "sequences"]:
                sequence_col = col
                break

        if sequence_col is None:
            raise ValueError(
                f"CSV file must have a 'sequence' or 'sequences' column. Found columns: {list(df.columns)}"
            )

        print(f"Reading sequences from CSV column: '{sequence_col}'")
        print(f"Found {len(df)} rows in CSV file")

        # Create a cycling iterator that repeats the sequences indefinitely
        sequences = []
        for _, row in df.iterrows():
            seq = row[sequence_col]
            if pd.notna(seq) and str(seq).strip():
                sequences.append(str(seq))

        # Cycle through sequences indefinitely
        import itertools

        return itertools.cycle(sequences)

    def _text_or_fasta_iterator(self, sequences_file: Path) -> Iterator[str]:
        """Iterator for text or FASTA files."""

        # Read all sequences into memory first
        sequences = []
        with open(sequences_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(">"):  # Skip FASTA headers
                    sequences.append(line)

        print(f"Found {len(sequences)} sequences in text/FASTA file")

        # Create a cycling iterator that repeats sequences indefinitely
        import itertools

        return itertools.cycle(sequences)

    def _preprocess_sequence(self, sequence: str) -> str:
        """Clean and validate DNA sequence."""
        # Remove whitespace and make uppercase
        sequence = sequence.strip().upper()

        # Replace invalid nucleotides with N
        sequence = re.sub(r"[^ATGCN]", "N", sequence)

        # Truncate if too long
        if len(sequence) > self.max_length - 2:  # Account for special tokens
            sequence = sequence[: self.max_length - 2]

        return sequence

    def __iter__(self):
        return self

    def __next__(self):
        """Return a batch of activations."""
        with t.no_grad():
            # If buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # Return a batch
            unreads = (
                (~self.read).nonzero().squeeze(1)
            )  # Only squeeze dim 1, keep dim 0
            if len(unreads) == 0:
                raise StopIteration("No more activations available")

            batch_size = min(self.out_batch_size, len(unreads))
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    def sequence_batch(self, batch_size: int = None) -> List[str]:
        """Get a batch of DNA sequences."""
        if batch_size is None:
            batch_size = self.refresh_batch_size

        sequences = []
        try:
            for _ in range(batch_size):
                seq = next(self.sequence_iterator)
                sequences.append(self._preprocess_sequence(seq))
        except StopIteration:
            if not sequences:
                raise StopIteration("End of sequence data reached") from None

        return sequences

    def tokenized_batch(self, batch_size: int = None):
        """Get a batch of tokenized DNA sequences."""
        sequences = self.sequence_batch(batch_size)
        if not sequences:
            raise StopIteration("No sequences available for tokenization")

        try:
            tokens = self.tokenizer(
                sequences,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )

            # Move to device
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            return tokens
        except Exception as e:
            print(f"Tokenization failed: {e}")
            print(f"Tokenizer pad_token: {self.tokenizer.pad_token}")
            print(f"Tokenizer eos_token: {self.tokenizer.eos_token}")
            raise

    def _standard_refresh(self):
        """Standard refresh method for regular buffers."""
        print("Refreshing activation buffer...")
        gc.collect()
        t.cuda.empty_cache()

        # Keep unread activations
        self.activations = self.activations[~self.read]
        current_idx = len(self.activations)

        # Create new buffer
        new_activations = t.empty(
            self.activation_buffer_size,
            self.d_model,
            device=self.device,
            dtype=self.model.dtype,
        )

        # Copy existing unread activations
        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        # Fill buffer with new activations
        initial_idx = current_idx

        # Create progress bar for buffer filling
        from tqdm import tqdm

        pbar = tqdm(
            total=self.activation_buffer_size - current_idx,
            desc="Generating activations",
            unit="acts",
            initial=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=1.0,  # Update every 1 second minimum
        )

        while current_idx < self.activation_buffer_size:
            try:
                with t.no_grad():
                    tokens = self.tokenized_batch()

                    # Use forward hooks approach (same as get_layer_activations in utils.py)
                    input_ids = tokens["input_ids"]
                    attention_mask = tokens["attention_mask"]

                    # Create a list to store activations
                    captured_activations = []

                    # Define a forward hook
                    def hook_fn(module, input, output):
                        captured_activations.append(output)

                    # Register the hook on the appropriate layer (MLP output)
                    # Note: self.model is LanguageModel wrapping the actual model
                    hook = self.model.esm.encoder.layer[
                        self.layer_idx
                    ].output.dense.register_forward_hook(hook_fn)

                    try:
                        # Perform a forward pass
                        with t.no_grad():
                            self.model(
                                input_ids=input_ids, attention_mask=attention_mask
                            )

                        # Get the activations
                        if captured_activations:
                            hidden_states = captured_activations[
                                0
                            ]  # First (and only) captured activation
                            # Ensure activations are on the correct device
                            hidden_states = hidden_states.to(self.device)
                        else:
                            raise RuntimeError("No activations captured by hook")
                    finally:
                        # Remove the hook
                        hook.remove()

                    # Apply attention mask to remove padding tokens (ensure on same device)
                    mask = attention_mask.to(self.device) != 0

                    # Remove special tokens if requested
                    if self.remove_special_tokens:
                        # Remove CLS (first token) and EOS/padding tokens
                        # NT format: [CLS] sequence [EOS] [PAD]...
                        mask = mask.clone()
                        mask[:, 0] = False  # Remove CLS token

                        # Remove EOS tokens (typically last non-padding token)
                        seq_lengths = mask.sum(dim=1)
                        for i, length in enumerate(seq_lengths):
                            if length > 0:
                                mask[i, length - 1] = False  # Remove EOS token

                    # Extract valid activations
                    hidden_states = hidden_states[mask]

                    # Add to buffer
                    remaining_space = self.activation_buffer_size - current_idx
                    if remaining_space <= 0:
                        break

                    n_to_add = min(len(hidden_states), remaining_space)
                    # Ensure hidden_states are on the correct device before indexing
                    hidden_states_on_device = hidden_states[:n_to_add].to(self.device)
                    self.activations[current_idx : current_idx + n_to_add] = (
                        hidden_states_on_device
                    )
                    current_idx += n_to_add

                    # Update progress bar
                    pbar.update(n_to_add)

            except StopIteration:
                pbar.set_description(f"End of data - buffer filled to {current_idx:,}")
                break

        # Close progress bar
        pbar.close()

        # Reset read flags
        self.read = t.zeros(current_idx, dtype=t.bool, device=self.device)

        # Trim buffer to actual size
        self.activations = self.activations[:current_idx]

        added_activations = current_idx - initial_idx
        print(
            f"✅ Buffer refreshed: +{added_activations:,} activations (total: {current_idx:,})"
        )

    # Create a difference buffer from two existing buffers
    @classmethod
    def create_difference_buffer(
        cls,
        base_buffer: "NTActivationBuffer",
        fine_tuned_buffer: "NTActivationBuffer",
    ) -> "NTActivationBuffer":
        """Create a difference activation buffer from two existing buffers."""
        if base_buffer.d_model != fine_tuned_buffer.d_model:
            raise ValueError(
                "Base and fine-tuned buffers must have the same activation dimension"
            )

        # Create a minimal buffer that will only store difference activations
        # We don't need to recreate the sequence iterator since we'll compute differences directly
        diff_buffer = object.__new__(cls)  # Create without __init__

        # Copy essential attributes
        diff_buffer.model_name = base_buffer.model_name
        diff_buffer.fine_tuned_weights = None
        diff_buffer.layer_idx = base_buffer.layer_idx
        diff_buffer.n_ctxs = base_buffer.n_ctxs
        diff_buffer.ctx_len = base_buffer.ctx_len
        diff_buffer.refresh_batch_size = base_buffer.refresh_batch_size
        diff_buffer.out_batch_size = base_buffer.out_batch_size
        diff_buffer.device = base_buffer.device
        diff_buffer.max_length = base_buffer.max_length
        diff_buffer.remove_special_tokens = base_buffer.remove_special_tokens
        diff_buffer.d_model = base_buffer.d_model
        diff_buffer.activation_buffer_size = base_buffer.activation_buffer_size

        # Store references to source buffers for dynamic computation
        diff_buffer.base_buffer = base_buffer
        diff_buffer.fine_tuned_buffer = fine_tuned_buffer

        # Initialize empty activations - will be computed on first refresh
        diff_buffer.activations = t.empty(
            0, diff_buffer.d_model, device=diff_buffer.device
        )
        diff_buffer.read = t.zeros(0, dtype=t.bool, device=diff_buffer.device)

        print(
            "✅ Created difference activation buffer (will compute differences on-demand)"
        )
        
        # Do initial refresh to fill the buffer properly
        print("  Performing initial difference buffer refresh...")
        diff_buffer.refresh()

        return diff_buffer

    def refresh_difference_buffer(self):
        """Refresh difference buffer by computing differences from full source buffers."""
        if not hasattr(self, "base_buffer") or not hasattr(self, "fine_tuned_buffer"):
            raise RuntimeError(
                "This is not a difference buffer - missing source buffer references"
            )

        # Only print every 10th refresh to reduce verbosity
        if not hasattr(self, '_refresh_count'):
            self._refresh_count = 0
        self._refresh_count += 1
        
        if self._refresh_count % 10 == 1:
            print(f"Refreshing difference activation buffer (#{self._refresh_count})...")
        elif self._refresh_count <= 2:
            print("Refreshing difference activation buffer...")

        # Ensure both source buffers are fully refreshed first
        if len(self.base_buffer.activations) < self.base_buffer.activation_buffer_size // 2:
            print("  Base buffer needs refresh...")
            self.base_buffer._standard_refresh()
        if len(self.fine_tuned_buffer.activations) < self.fine_tuned_buffer.activation_buffer_size // 2:
            print("  Fine-tuned buffer needs refresh...")
            self.fine_tuned_buffer._standard_refresh()

        # Get the full activations from both buffers (not just one batch)
        base_acts = self.base_buffer.activations
        ft_acts = self.fine_tuned_buffer.activations

        # Match sizes
        n_acts = min(len(base_acts), len(ft_acts))

        # Compute differences for the full buffer
        diff_activations = ft_acts[:n_acts] - base_acts[:n_acts]

        # Store new difference activations
        self.activations = diff_activations
        self.read = t.zeros(n_acts, dtype=t.bool, device=self.device)

        if self._refresh_count % 10 == 1 or self._refresh_count <= 2:
            print(f"✅ Difference buffer refreshed with {n_acts:,} activations")

    def refresh(self):
        """Override refresh method for difference buffers."""
        if hasattr(self, "base_buffer") and hasattr(self, "fine_tuned_buffer"):
            # This is a difference buffer - use special refresh logic
            self.refresh_difference_buffer()
        else:
            # This is a regular buffer - use normal refresh logic
            self._standard_refresh()

    @property
    def config(self):
        """Return configuration for saving/loading."""
        return {
            "model_name": self.model_name,
            "layer_idx": self.layer_idx,
            "d_model": self.d_model,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
            "max_length": self.max_length,
            "remove_special_tokens": self.remove_special_tokens,
        }


# Create a simple DataLoader-like wrapper for the activation buffer
class NTActivationDataLoader:
    """DataLoader wrapper for NTActivationBuffer to integrate with existing training code."""

    def __init__(self, activation_buffer: NTActivationBuffer):
        self.activation_buffer = activation_buffer
        self.d_model = activation_buffer.d_model

    def __iter__(self):
        return self.activation_buffer

    def __next__(self):
        return next(self.activation_buffer)
