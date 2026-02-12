# UPS Challenge Baselines

Reference implementation for loading data from the [Unsupervised People's Speech (UPS) Challenge](https://mlcommons.org/datasets/peoples-speech/) dataset.

## Overview

This repository provides a WebDataset-based dataloader for efficiently streaming and processing audio data from the UPS dataset. It includes support for language-specific filtering using soft language ID markers.

## Structure

```
ups_challenge/
├── dataloaders/
│   ├── base.py           # Core dataloader and preprocessing
│   ├── urls.py           # URL building and language filtering
│   └── build_index.py    # Language ID index construction
└── examples/
    └── main.py           # Usage example with wav2vec2
```

## Features

- **Streaming data loading** via WebDataset from HuggingFace
- **Language-specific filtering** using pre-computed language ID indices
- **Audio preprocessing**: resampling, mono conversion, chunking
- **Random chunk sampling** for long audio files
- **Automatic fallback**: loads first 4 tar files if no languages specified

## Usage

### Basic Example

```python
from ups_challenge.dataloaders.base import build_wds_dataset, collate_fn
import torch

# Load specific languages (e.g., English)
wds_dataset = build_wds_dataset(langs=['en'])

# Or load all languages (uses first 4 tar files by default)
wds_dataset = build_wds_dataset(langs=[])

data_loader = torch.utils.data.DataLoader(
    wds_dataset,
    batch_size=1,
    num_workers=1,
    collate_fn=collate_fn,
)

for batch in data_loader:
    if batch is None:
        continue
    # batch contains 'input_values' and 'attention_mask'
    print(batch['input_values'].shape)
```

### Running the Example

From the project root:

```bash
python -m ups_challenge.examples.main
```

## Requirements

- `torch`
- `transformers`
- `webdataset`
- `torchcodec`
- `braceexpand`
- `tqdm`

Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```

## Language Filtering

The dataloader supports filtering by language codes (e.g., `'en'`, `'es'`, `'fr'`). When languages are specified:

1. A language ID index is built (or loaded from `./data/lid_index.pkl`)
2. Only tar files containing the specified languages are streamed
3. This significantly reduces download time and storage requirements

Without language specification, the loader defaults to the first 4 tar files for quick testing.

## Audio Processing

- **Target sample rate**: 16kHz
- **Chunk duration**: 10 seconds
- **Max chunks per file**: 16 (for long files)
- **Output format**: `[N_chunks, samples]` tensor with attention mask

Short files (<10s) are padded to chunk length; long files are randomly sampled.
