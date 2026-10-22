# MulEG
[EMNLP2026 Findings] 
MulEG: Multi-Level Explanation-Guided Feature Attribute Graph for Zero-Shot Tabular Classification


## Overview

Zero-shot classification of structured tabular data aims to recognize classes that are unseen during training by leveraging textual explanations as supervision. Prior methods either flatten tabular data into explanation sequences (disrupting the row-column topology) or model relations with fixed logical operators (overlooking learnable dependencies among features).

**MulEG** addresses both limitations by jointly modeling:

- **Multi-level textual explanations** (sample-, attribute-, and label-level) that preserve the intrinsic row-column topological structure of tables;
- **A feature attribute graph** with a differentiable message-passing mechanism that captures fine-grained dependencies among heterogeneous features;
- A **Structural Causal Model** inspired view linking tabular features and target labels.

Experiments on sixteen tabular tasks demonstrate that MulEG significantly outperforms state-of-the-art baselines across semantic, mixed, numerical, ordinal, and symbolic domains.

## Key Contributions

1. Multi-level textual explanation modeling that preserves the row-column structure of tabular data.
2. A differentiable message-passing feature attribute graph for capturing feature dependencies.
3. Extensive experiments, ablations, and faithfulness analysis on sixteen zero-shot tabular tasks.

## Repository Structure

```
.
├── models/
│   ├── causal_gnn/          # GNN wrapper, hybrid encoder, symbolic encoder, losses
│   ├── causal_graph/        # Task-level DAG construction and regularization
│   └── guided_bert_clues.py # Explanation-guided BERT encoder
├── experiments/
│   ├── clues_53gai.py       # Main experiment script (zero-shot tabular classification)
│   └── trainer.py           # Custom HF trainers
├── clues_data/              # Datasets, task-level DAGs, and label details
├── guidelines/              # Guideline generation utilities
```

## Requirements

- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- NumPy, scikit-learn, tqdm

A `bert-base-uncased` checkpoint is used as the default backbone (`--model_name_or_path bert-base-uncased`).

## Quick Start

1. Clone the repository and install dependencies.

2. Prepare the data under `clues_data/` (task-level DAGs and label details).

3. Run the main experiment:

```bash
python experiments/clues_53gai.py \
    --model_name_or_path bert-base-uncased \
    --output_dir ./logs/run_muleg
```


```
## License
This project is released for research purposes.
