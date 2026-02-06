import copy
import queue
import threading
from typing import Any, Dict, List, Tuple

import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from .autoencoder_models import AutoEncoder
except ImportError:
    try:
        from autoencoder_models import AutoEncoder
    except ImportError:
        AutoEncoder = None  # Will handle gracefully if needed


def load_model_and_tokenizer(
    fine_tuned_weights: str = None,
    model_size: str = "50m",
) -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Load the pre-trained model and tokenizer.

    Args:
        fine_tuned_weights: Path to fine-tuned weights file to load into base model
        model_size: Model size (e.g., "50m", "100m", "250m", "500m")
    """

    # Construct model name from size parameter
    model_name = f"InstaDeepAI/nucleotide-transformer-v2-{model_size}-multi-species"
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Load fine-tuned weights if provided
    if fine_tuned_weights:
        print(f"Loading fine-tuned weights from: {fine_tuned_weights}")

        # Create deep copy of original state dict for comparison (only the state dict, not the whole model)
        original_state_dict = copy.deepcopy(model.state_dict())

        # Load fine-tuned weights directly into the model
        checkpoint = torch.load(
            fine_tuned_weights, map_location="cpu", weights_only=True
        )

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            fine_tuned_state_dict = checkpoint["model_state_dict"]
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        elif isinstance(checkpoint, dict) and any(
            key.startswith(("esm.", "encoder.", "embeddings.")) for key in checkpoint
        ):
            # Direct state dict format
            fine_tuned_state_dict = checkpoint
        else:
            raise ValueError(
                f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())}"
            )

        model.load_state_dict(fine_tuned_state_dict, strict=False)

        # Verify weights have actually changed
        current_state_dict = model.state_dict()
        weights_changed = False
        changed_layers = []

        for key in original_state_dict:
            if key in current_state_dict and not torch.equal(
                original_state_dict[key], current_state_dict[key]
            ):
                weights_changed = True
                changed_layers.append(key)

        if weights_changed:
            print(
                f"Fine-tuned weights loaded successfully. {len(changed_layers)} layers updated."
            )

        else:
            print(
                "WARNING: Fine-tuned weights loaded but no parameters appear to have changed!"
            )
            print(
                "This might indicate that the weights file contains the same weights as the base model."
            )

    return model, tokenizer


def load_sae_model(
    sae_model_path: str = "weights/extra_addgene_pretraining/SAE_NT50_plasmidpre_L12_mlpout_40mtokens_190325.pt",
    device: str = "cuda",
) -> nn.Module:
    """
    Load a pre-trained SAE model from a file.

    Args:
        sae_model_path (str): Path to the saved SAE model file.
        device (str): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        nn.Module: The loaded SAE model.
    """
    model, tokenizer = load_model_and_tokenizer()
    cfg = {
        "seed": 49,
        "batch_size": 4096,
        "buffer_mult": 384,
        "lr": 5e-5,
        "num_tokens": tokenizer.vocab_size,
        "d_model": model.config.hidden_size,
        "l1_coeff": 1e-1,
        "l0_coeff": 1,
        "beta1": 0.9,
        "beta2": 0.999,
        "dict_mult": 8,  # hidden_d = d_model * dict_mult
        "seq_len": 512,
        "d_mlp": model.config.hidden_size,
        "remove_rare_dir": False,
        "total_training_steps": 10000,
        "lr_warm_up_steps": 1000,
        "device": device,
        "tempterature": 0.05,
        "activation_treshold": 0.3,
        "enc_dtype": torch.float32,
    }
    sae_model = AutoEncoder(cfg)
    weights = torch.load(sae_model_path, map_location=device, weights_only=True)
    if "model_state_dict" in weights:
        weights = weights["model_state_dict"]

    sae_model.load_state_dict(weights)
    sae_model.to(device)
    return sae_model


def load_validation_set(val_set_idx: int) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Load a validation dataset from pickle files.

    Args:
        val_set_idx (int): Index of the validation set to load (0, 1, or 2).

    Returns:
        Tuple[torch.Tensor, pd.DataFrame]: Tokens and token dataframe
    """
    folder_path = "annotated_seqs/"

    # Load tokenizer
    _, tokenizer = load_model_and_tokenizer()

    ann_file = f"{folder_path}/ann_of_1000_seqs_set{val_set_idx}.csv"

    # Process annotation file
    df_val = pd.read_csv("data/blast_geac_ext_169k_val_random.csv")
    df_annotated = load_and_process_annotations(ann_file)
    tokens, _ = extract_and_tokenize_sequences(df_annotated, df_val, tokenizer)

    token_df = pd.read_pickle(
        folder_path + f"token_df_1k_ss{val_set_idx}_standardized.pkl"
    )
    return tokens, token_df


def load_and_process_annotations(file_path):
    """Load CSV and add 'valseq_' prefix to seq_id column if not already present."""
    df = pd.read_csv(file_path)
    df["seq_id"] = df["seq_id"].astype(str)
    # Add 'valseq_' prefix only if it's not already there
    df["seq_id"] = df["seq_id"].apply(
        lambda x: x if x.startswith("valseq_") else f"valseq_{x}"
    )
    return df


def extract_and_tokenize_sequences(df_annotations, df_val, tokenizer):
    """Extract sequence IDs, get corresponding sequences, and tokenize them."""
    # Extract and sort sequence IDs
    seq_ids = list(set(df_annotations["seq_id"]))
    # More robust parsing of sequence IDs
    parsed_ids = []
    for seq_id in seq_ids:
        try:
            if "valseq_" in seq_id:
                parsed_ids.append(int(seq_id.split("valseq_")[1]))
            else:
                parsed_ids.append(int(seq_id))
        except ValueError:
            print(f"Warning: Could not parse seq_id: {seq_id}")
            continue

    seq_ids = sorted(parsed_ids)

    # Get and tokenize sequences
    sequences = df_val["sequence"].iloc[seq_ids].tolist()
    tokens = tokenizer(
        sequences,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return tokens, seq_ids


##### ACTIVATION EXTRACTION


def get_layer_activations(
    model, input_ids, attention_mask=None, layer_N=11, token_level=True, model_name="NT"
) -> List[torch.Tensor]:
    """
    Extract activations from a specific layer of the model.
    Args:
        model: The model from which to extract activations.
        input_ids: Input tensor of token IDs.
        attention_mask: Optional attention mask tensor.
        layer_N: The index of the layer from which to extract activations.
        token_level: If True, return activations at the token level, otherwise
        return aggregate over entire sequence.
        model_name: The name of the model architecture
    Returns:
        activations: A list of activations from the specified layer.
    """

    # Ensure the model is in evaluation mode
    model.eval()

    # Create a list to store activations
    activations = []

    # Define a forward hook
    def hook_fn(module, input, output):
        if token_level:
            # Return token-level activations (keep all sequence positions)
            activations.append(output)
        else:
            # Return sequence-level aggregate activations
            # Apply attention mask if provided for proper aggregation
            if attention_mask is not None:
                # Expand attention mask to match activation dimensions
                mask_expanded = attention_mask.unsqueeze(-1).expand(output.size())
                # Mask out padding tokens
                masked_output = output * mask_expanded
                # Sum over sequence length, then divide by number of valid tokens
                valid_token_count = attention_mask.sum(dim=1, keepdim=True)
                # Avoid division by zero
                valid_token_count = torch.clamp(valid_token_count, min=1)
                aggregated = masked_output.sum(dim=1) / valid_token_count
            else:
                # Simple mean pooling over sequence length
                aggregated = output.mean(dim=1)
            activations.append(aggregated)

    # Register the hook on appropriate layers
    hooks = []
    if model_name == "NT":
        hooks.append(
            model.esm.encoder.layer[layer_N].output.dense.register_forward_hook(hook_fn)
        )

    elif model_name == "metagene-1":
        hooks.append(
            model.model.layers[layer_N].input_layernorm.register_forward_hook(hook_fn)
        )

    # Perform a forward pass
    with torch.no_grad():
        if attention_mask is not None:
            model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            model(input_ids=input_ids)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return activations


def get_residual_activations(
    model, input_ids, attention_mask=None, layer_N=11, position="post_mlp"
):
    model.eval()
    activations = []

    def hook_fn(module, input, output):
        activations.append(input[0])

    hooks = []
    if position == "pre_mlp":
        hooks.append(
            model.esm.encoder.layer[layer_N].intermediate.register_forward_hook(hook_fn)
        )
    elif position == "post_mlp":
        # Hook after the entire layer to get full residual stream
        hooks.append(model.esm.encoder.layer[layer_N].register_forward_hook(hook_fn))

    with torch.no_grad():
        (
            model(input_ids=input_ids, attention_mask=attention_mask)
            if attention_mask is not None
            else model(input_ids=input_ids)
        )

    for hook in hooks:
        hook.remove()

    return activations


def extract_layer_activations_and_latents(
    model_nt, sae_model, tokens, layer_num, batch_size=32, device="cuda"
):
    """
    Extract MLP activations and corresponding SAE latent representations.
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
        for i in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_seqs)

            # Reshape tokens for current batch
            batch_input_ids = tokens["input_ids"][start_idx:end_idx].to(device)
            batch_attention_mask = tokens["attention_mask"][start_idx:end_idx].to(
                device
            )

            with torch.no_grad(), autocast():
                # Get MLP activations
                try:
                    mlp_act = get_layer_activations(
                        model_nt,
                        batch_input_ids,
                        batch_attention_mask,
                        layer_N=layer_num,
                    )

                    # Additional error checking
                    if mlp_act is None or len(mlp_act) == 0 or mlp_act[0].numel() == 0:
                        print(f"No or empty activations retrieved for batch {i}")
                        continue

                    mlp_act = mlp_act[0].reshape(-1, d_mlp)

                    # Check for empty tensors after reshaping
                    if mlp_act.numel() == 0:
                        print(f"Empty tensor after reshaping in batch {i}")
                        continue

                    # Forward pass through SAE
                    (
                        encoded_acts,
                        _,
                        _,
                    ) = sae_model.encode(mlp_act, return_active=True)

                    # Check latents before appending
                    if encoded_acts.numel() == 0:
                        print(f"Empty latents in batch {i}")
                        continue

                    all_latents.append(encoded_acts)
                    all_acts.append(mlp_act)

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


##### ANNOTATION


def list_flatten(nested_list) -> list:
    """
    Flattens a list of lists
    """
    return [x for y in nested_list for x in y]


def get_seq_annotation(
    token_spec: Dict[str, Any],
    df_annotated: pd.DataFrame,
    special_tokens: List[str],
    descriptor_col: str,
) -> List[str]:
    """
    Get the annotation(s) for a given token, including special tokens and regular annotations.

    Args:
    token_spec (Dict[str, Any]): A dictionary specifying the token, with keys 'seq_id', 'start', 'end', and 'token'.
    df_annotated (pd.DataFrame): A DataFrame containing the annotations.
    special_tokens (List[str]): A list of special tokens to check for.

    Returns:
    List[str]: A list of annotations that overlap with the specified token, including special token annotations.
    """
    required_columns = ["seq_id", "qstart", "qend"]
    if not all(col in df_annotated.columns for col in required_columns):
        raise ValueError(
            f"DataFrame is missing one or more required columns: {required_columns}"
        )

    if token_spec["seq_id"] not in df_annotated["seq_id"].unique():
        raise ValueError(
            f"Sequence ID '{token_spec['seq_id']}' not found in the DataFrame."
        )

    if token_spec["start"] >= token_spec["end"]:
        raise ValueError("Start location must be less than end location.")

    if "token" not in token_spec:
        raise ValueError("Token specification must include 'token' key.")

    token_annotations = []

    # Check if it's a special token
    if token_spec["token"] in special_tokens:
        token_annotations.append(f"special token: {token_spec['token']}")
        return token_annotations, None, None

    # Always check for overlapping annotations, regardless of whether it's a special token or not

    # to-do: make sure (1) we want 'qstart/qend' and (2) any overlap should be enough

    annotated_regions = df_annotated[
        (df_annotated["seq_id"] == token_spec["seq_id"])
        & (df_annotated["qstart"] < token_spec["end"])
        & (df_annotated["qend"] > token_spec["start"])
    ]
    token_annotations.extend(
        annotated_regions[descriptor_col].tolist()
    )  ## choice: could also use 'Description' for more detail

    # additionally provide information about quality of the annotation
    evalue_match = annotated_regions["evalue"].values
    pident_match = annotated_regions["pident"].values

    return token_annotations, evalue_match, pident_match


def create_context(
    batch: List[str], position: int, len_prefix: int, len_suffix: int
) -> str:
    """
    Create a context string for a token with specified prefix and suffix lengths.

    Args:
        batch (List[str]): The list of tokens.
        position (int): The position of the current token.
        len_prefix (int): The desired length of the prefix.
        len_suffix (int): The desired length of the suffix.

    Returns:
        str: A formatted context string.
    """
    prefix_start = max(0, position - len_prefix)
    suffix_end = min(len(batch), position + 1 + len_suffix)

    prefix = "".join(batch[prefix_start:position])
    current_token = batch[position]
    suffix = "".join(batch[position + 1 : suffix_end])

    return f"{prefix} |{current_token}| {suffix}"


def make_token_df_new(
    tokens: torch.Tensor,
    tokenizer,
    df_annotated: pd.DataFrame,
    seq_ids: List[str],
    len_prefix: int = 5,
    len_suffix: int = 2,
    nucleotides_per_token: int = 6,
    descriptor_col: str = "Feature",
) -> pd.DataFrame:
    """
    Create a DataFrame with token information, context, and annotations for batched input.
    Includes progress bars for batch and token processing.

    Args:
        tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
        tokenizer: The tokenizer object.
        df_annotated (pd.DataFrame): DataFrame with sequence annotations.
        seq_ids (List[str]): List of sequence identifiers for each batch item.
        len_prefix (int): Length of context prefix.
        len_suffix (int): Length of context suffix.
        nucleotides_per_token (int): Number of nucleotides represented by each token.

    Returns:
        pd.DataFrame: DataFrame containing token information and annotations.
    """
    if tokens.numel() == 0:
        return pd.DataFrame(
            columns=[
                "seq_id",
                "tokens",
                "context",
                "token_annotations",
                "context_annotations",
            ]
        )

    batch_size, seq_len = tokens.shape
    special_tokens = tokenizer.all_special_tokens
    data = []

    # Main progress bar for batches
    batch_progress = tqdm(range(batch_size), desc="Processing batches", unit="batch")

    for batch_idx in batch_progress:
        seq_tokens = tokens[batch_idx]
        seq_id = seq_ids[batch_idx]

        # Decode each token id to the corresponding string
        str_tokens = [tokenizer.decode(token).replace(" ", "") for token in seq_tokens]

        for position, token in enumerate(str_tokens):
            token_start = position * nucleotides_per_token
            token_end = token_start + nucleotides_per_token

            # Token annotation
            token_spec = {
                "seq_id": f"valseq_{seq_id}",
                "start": token_start,
                "end": token_end,
                "token": token,
            }
            token_annotation, evalue_match, pident_match = get_seq_annotation(
                token_spec, df_annotated, special_tokens, descriptor_col
            )

            # Context and its annotation
            context = create_context(str_tokens, position, len_prefix, len_suffix)
            context_start = max(0, token_start - len_prefix * nucleotides_per_token)
            context_end = min(
                len(str_tokens) * nucleotides_per_token,
                token_end + len_suffix * nucleotides_per_token,
            )
            context_spec = {
                "seq_id": f"valseq_{seq_id}",
                "start": context_start,
                "end": context_end,
                "token": context,
            }

            context_annotation, _, _ = get_seq_annotation(
                context_spec, df_annotated, special_tokens, descriptor_col
            )

            data.append(
                {
                    "seq_id": seq_id,
                    "token_pos": position,
                    "tokens": token,
                    "context": context,
                    "token_annotations": token_annotation,
                    "context_annotations": context_annotation,
                    "e-value annotation": evalue_match,
                    "percentage match": pident_match,
                }
            )

        # Update batch progress description
        batch_progress.set_description(f"Completed batch {batch_idx + 1}/{batch_size}")

    return pd.DataFrame(data)


#### Handle addgene data

def preprocess_data(train_data_path: str, test_data_path: str, min_length: int = 0):
    """Process the data."""
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    x_train, y_train = train_data[["sequence"]], train_data["nations"]
    y_test = test_data["nations"]
    x_test = test_data[["sequence"]]

    # Combine labels from train and test datasets
    processed_labels = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    unique_labels = processed_labels.unique()
    label_to_int = {label: int(i) for i, label in enumerate(unique_labels)}

    # map labels to integers
    y_train = y_train.map(label_to_int)
    y_test = y_test.map(label_to_int)

    print(f"y_test shape: {y_test.shape}")

    # reset indices before concat
    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    df_train = pd.concat([x_train, y_train], axis=1)
    df_val = pd.concat([x_test, y_test], axis=1)

    # Filter out sequences shorter than min_length and clean them
    min_length = 0
    df_train = df_train[df_train["sequence"].str.len() > min_length]
    df_val = df_val[df_val["sequence"].str.len() > min_length]

    print(f"test_data shape: {test_data.shape}")

    # Ensure indices are reset correctly
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    return df_train, df_val


#### Activation Storage


class ActivationsDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, cache_size=10000):
        self.h5_path = h5_path
        self.cache_size = cache_size
        self.cache = {}

        # Read metadata once
        with h5py.File(h5_path, "r") as f:
            self.length = f["activations"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Read data in blocks to minimize file operations
        block_start = (idx // self.cache_size) * self.cache_size
        block_end = min(block_start + self.cache_size, self.length)

        with h5py.File(self.h5_path, "r") as f:
            block_data = f["activations"][block_start:block_end]

        # Cache the block
        for i, data in enumerate(block_data):
            cache_idx = block_start + i
            self.cache[cache_idx] = torch.FloatTensor(data)

        return self.cache[idx]


class BackgroundLoader:
    def __init__(self, h5_file, chunk_size):
        self.h5_file = h5_file
        self.chunk_size = chunk_size
        self.queue = queue.Queue(maxsize=2)  # Keep 2 chunks in memory
        self.thread = None
        self.current_chunk_idx = 0

    def load_chunk(self, chunk_idx):
        data = self.h5_file["activations"]
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(data))
        return torch.from_numpy(data[start_idx:end_idx][:]).float()

    def loader_thread(self, next_chunk_idx):
        chunk = self.load_chunk(next_chunk_idx)
        self.queue.put((next_chunk_idx, chunk))

    def get_next_chunk(self):
        # Start loading next chunk in background if not already loading
        if self.thread is None or not self.thread.is_alive():
            next_chunk_idx = self.current_chunk_idx + 1
            self.thread = threading.Thread(
                target=self.loader_thread, args=(next_chunk_idx,)
            )
            self.thread.start()

        # Get current chunk
        chunk_idx, chunk = self.queue.get()
        self.current_chunk_idx = chunk_idx
        return chunk


class ChunkedActivationsDataset(Dataset):
    def __init__(
        self, h5_path, batch_size=2048 * 4, chunks_in_memory=4, max_chunks=None
    ):
        self.h5_file = h5py.File(h5_path, "r")
        self.data = self.h5_file["activations"]
        self.length = self.data.shape[0]
        self.batch_size = batch_size

        # Make chunk size a multiple of batch_size
        self.chunk_size = chunks_in_memory * batch_size

        # Adjust length if max_chunks is specified
        if max_chunks:
            self.length = min(self.length, max_chunks * self.chunk_size)

        # Initialize loader
        self.loader = BackgroundLoader(self.h5_file, self.chunk_size)
        # Initialize first chunk
        self.current_chunk = self.loader.load_chunk(0)
        self.current_chunk_idx = 0
        self.loader.get_next_chunk()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        if chunk_idx != self.current_chunk_idx:
            self.current_chunk = self.loader.get_next_chunk()
            self.current_chunk_idx = chunk_idx

        local_idx = idx % self.chunk_size
        return self.current_chunk[local_idx]

    def __del__(self):
        self.h5_file.close()
