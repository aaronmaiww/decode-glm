# Decode-gLM: Tools to Interpret, Audit, and Steer Genomic Language Models 

This repository contains the essential code for reproducing key results from our paper **Decode-gLM: Tools to Interpret, Audit, and Steer Genomic Language Models**.

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/aaronmaiww/decode-glm.git
cd decode-glm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Request Access to Datasets

The datasets required for reproduction contain proprietary Addgene plasmid sequences and are available to qualified researchers upon request. Please contact [contact information] with:
1. Brief description of research purpose  
2. Institutional affiliation
3. Agreement to use data solely for research purposes

### 4. Download Pre-trained Models (Optional)

We host our trained SAE weights on the Hugging Face Model Hub:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download

# Download SAE for NT-50M layer 1
local = hf_hub_download(
    repo_id="mawairon/sae-nt50-l1",
    filename="sae_nt50base_L1/checkpoint.pt"
)
print("Downloaded to", local)
```

**Hardware Requirements**: GPU recommended for training (≥8GB VRAM). CPU-only mode supported but significantly slower.

---

## 📊 Key Results Reproduction

This repository enables reproduction of four key results from our paper:

### Result 1: SAE Biological Concept Discovery
**Main Result**: SAEs trained on genomic language models discover interpretable biological concepts with high F1 scores.

```bash
# Evaluate biological concept discovery on trained SAE
python scripts/get_f1_scores_batchtopk_minimal.py \
    --model-type batchtopk \
    --sae-model-path /path/to/trained/model.pt \
    --activation-dim 512 \
    --dict-size 4096 \
    --k 32
```

**Expected Output**: F1 scores for biological annotations showing concept discovery performance.

### Result 2: Model Steering with SAE Features
**Main Result**: SAE latents can steer genomic language models to generate sequences with specific biological properties.

```bash
# Run steering experiments (reproduces steering results CSV)
jupyter notebook notebooks/Steering_AMR_notebook_.ipynb
```

**Key Files**:
- `data/steering_results/fixed_activation_steering.csv` - Pre-computed steering results
- `src/steering_utils_pj.py` - Core steering functions
- Expected runtime: 24-72 minutes on GPU for full experiment

### Result 3: Meta-SAE for Cross-Feature Analysis  
**Main Result**: Meta-SAEs trained on SAE decoder weights show latent specialised on HIV.

```bash
# Train and evaluate meta-SAE for HIV probe
jupyter notebook notebooks/metaSAE_eval_HIVprobe.ipynb
```

**Key Files**:
- `data/meta_SAE/261025_probe_HIV_results_summary.csv` - HIV probe results  
- `notebooks/meta_SAE.ipynb` - Meta-SAE training workflow
- `src/train_probes.py` - Linear probe training utilities

### Result 4: BLAST Analysis of Training Data
**Main Result**: CMV enhancer sequences are present in the nucleotide transformer pretraining corpus.

**Setup**: First download the nucleotide transformer pretraining corpus:
1. **Download the nucleotide transformer pretraining corpus** (genomic .fna files)
2. **Place all .fna files** in the same directory as the BLAST script

```bash
# Run BLAST search against pretraining data
cd scripts/
./batch_blast_csv.sh ../data/query_sequences/CMV_enhancer.fasta
```

**Key Files**:
- `data/blast_results_cmvenhancers/` - Example BLAST results (8,753 hits across 5 species)
- `scripts/batch_blast_csv.sh` - BLAST search script
- Processing: Handles thousands of genomic .fna files in memory-efficient batches

---

## 📁 Repository Structure

```
├── scripts/
│   ├── train_sae_hyperparameter_sweep.py    # Train SAE models with hyperparameter sweeps
│   ├── get_f1_scores_batchtopk_minimal.py   # Evaluate biological concept discovery
│   └── batch_blast_csv.sh                   # BLAST search against pretraining corpus
├── src/
│   ├── measure_monosemanticity.py           # Core F1 calculation and concept detection
│   ├── utils.py                             # Model loading and data utilities  
│   ├── steering_utils_pj.py                 # Model steering functions
│   ├── train_probes.py                      # Linear probe training utilities
│   ├── interplm_training/                   # SAE training framework
│   └── interplm_sae/                        # SAE model definitions (BatchTopKSAE)
├── notebooks/
│   ├── Steering_AMR_notebook_.ipynb         # Model steering experiments
│   ├── metaSAE_eval_HIVprobe.ipynb         # Meta-SAE HIV probe evaluation
│   ├── meta_SAE.ipynb                       # Meta-SAE training workflow
│   └── fig2_main_plots.ipynb               # Figure 2 reproduction
├── data/
│   ├── steering_results/                    # Pre-computed steering experiment results
│   ├── meta_SAE/                           # Meta-SAE weights and HIV probe results
│   ├── blast_results_cmvenhancers/         # Example BLAST search results
│   └── sae_latent_eval 					# Annotations associated with SAE latents
└── requirements.txt                         # Python dependencies
```

## 🔧 Advanced Configuration

### SAE Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Model size | Required | 50, 100, 250, 500, 2500 |
| `--layer` | Layer number | Required | 0-23 (depending on model) |
| `--sequences-file` | Path to DNA sequences | Required | CSV/FASTA/TXT file |
| `--max-combinations` | Limit hyperparameter combinations | All | Integer |
| `--use-wandb` | Enable experiment tracking | False | Flag |

### F1 Evaluation Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model-type` | SAE architecture | `batchtopk` | `batchtopk`, `identity` |
| `--sae-model-path` | Trained SAE weights | Required | `.pt` file path |
| `--activation-dim` | Model activation dimension | 512 | Integer |
| `--dict-size` | Number of SAE features | 4096 | Integer |
| `--k` | BatchTopK parameter | 32 | Integer |
| `--f1-type` | F1 calculation method | `domain` | `domain`, `standard` |

### Optional: Train Your Own SAE Models

```bash
# Train SAE on NT-50M layer 6 with default hyperparameters
python scripts/train_sae_hyperparameter_sweep.py \
    --model 50 \
    --layer 6 \
    --sequences-file /path/to/your/sequences.csv \
    --use-wandb \
    --wandb-entity your-entity
```


## 📚 Citation

If you use this code, please cite our paper:

```bibtex
@article {Maiwald2025.10.31.685860,
	author = {Maiwald, Aaron and Jedryszek, Piotr and Draye, Florent and Scholkopf, Bernhard and Morris, Garrett M. and Crook, Oliver M.},
	title = {Decode-gLM: Tools to Interpret, Audit, and Steer Genomic Language Models},
	elocation-id = {2025.10.31.685860},
	year = {2026},
	doi = {10.1101/2025.10.31.685860},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/01/25/2025.10.31.685860},
	eprint = {https://www.biorxiv.org/content/early/2026/01/25/2025.10.31.685860.full.pdf},
	journal = {bioRxiv}
}
```

#

## Data Availability Statement

The training and validation datasets contain proprietary Addgene plasmid sequences and are available to qualified researchers upon request. Please contact [contact information] with:
1. Brief description of research purpose
2. Institutional affiliation  
3. Agreement to use data solely for research purposes

---

**Hardware Requirements**: GPU recommended for training (>=8GB VRAM). CPU-only mode supported but significantly slower.
