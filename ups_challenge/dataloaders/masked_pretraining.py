"""Dataloader for HuBERT masked pre-training.

Reads a pre-built index (from prepare_pretraining_index.py) that maps
(tar_number, key) -> {start_sec, end_sec, labels}.  Streams audio from
HuggingFace tars via WebDataset, extracts the specific chunk for each
matching file, and returns (waveform, labels) pairs.
"""

import os
import pickle
import re

import numpy as np
import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder


def _build_tar_urls(tar_numbers, hf_token):
    """Build pipe:curl URLs for a specific set of tar numbers."""
    token = f"Authorization:Bearer {hf_token}"
    urls = []
    for tn in tar_numbers:
        folder = "audio" if int(tn) <= 5000 else "audio2"
        raw = (
            f"https://huggingface.co/datasets/MLCommons/"
            f"unsupervised_peoples_speech/resolve/main/{folder}/{tn}.tar?download=True"
        )
        urls.append(f"pipe:curl -s -L {raw} -H {token}")
    return urls


def _decode_pretraining(sample, lookup, target_sr=16000):
    """Decode a sample and pair with pre-computed labels if it is in the index.

    Returns list of dicts with 'waveform' (np.ndarray) and 'labels' (np.ndarray).
    """
    mp3_bytes, key, url = sample

    # Extract tar number from URL
    m = re.search(r"/(\d+)\.tar", url)
    if m is None:
        return []
    tar_number = m.group(1)

    entries = lookup.get((tar_number, key))
    if entries is None:
        return []

    results = []
    try:
        decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)
        
        for entry in entries:
            waveform = decoder.get_samples_played_in_range(
                entry["start_sec"], entry["end_sec"]
            ).data.squeeze(0)
            
            results.append({
                "waveform": waveform.numpy(),
                "labels": entry["labels"],
            })
            
    except Exception:
        return []

    return results


def collate_pretraining(batch):
    """Stack waveforms -> [B, T], pad labels -> [B, max_frames] (pad=-100)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    waveforms = [torch.from_numpy(b["waveform"]) for b in batch]
    labels = [torch.from_numpy(b["labels"].astype(np.int64)) for b in batch]

    # Pad waveforms
    max_wav_len = max(w.shape[0] for w in waveforms)
    padded_wav = torch.zeros(len(waveforms), max_wav_len)
    attention_mask = torch.zeros(len(waveforms), max_wav_len, dtype=torch.long)
    for i, w in enumerate(waveforms):
        padded_wav[i, : w.shape[0]] = w
        attention_mask[i, : w.shape[0]] = 1

    # Pad labels (use -100 for ignore_index in cross-entropy)
    max_label_len = max(l.shape[0] for l in labels)
    padded_labels = torch.full((len(labels), max_label_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, : l.shape[0]] = l

    return {
        "waveform": padded_wav,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }


def build_pretraining_dataset(index_path, hf_token=None, target_sr=16000):
    """Build a WebDataset that yields {waveform, labels} from a pretraining index.

    Args:
        index_path: Path to the pickle index produced by prepare_pretraining_index.py.
        hf_token: HuggingFace token (falls back to HF_TOKEN env var).
        target_sr: Target sample rate for audio decoding.
    """
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    with open(index_path, "rb") as f:
        index_entries = pickle.load(f)

    # Build lookup: (tar_number, key) -> list of entries
    lookup = {}
    tar_numbers = set()
    for entry in index_entries:
        k = (entry["tar_number"], entry["key"])
        if k not in lookup:
            lookup[k] = []
        lookup[k].append(entry)
        tar_numbers.add(entry["tar_number"])

    urls = _build_tar_urls(tar_numbers, hf_token)

    # Helper to flatten list of lists
    def flatten_list(stream):
        for sample in stream:
            if isinstance(sample, list):
                for x in sample:
                    yield x
            else:
                yield sample

    dataset = (
        wds.WebDataset(urls, shardshuffle=True, handler=wds.handlers.ignore_and_continue)
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
        .map(lambda s: _decode_pretraining(s, lookup, target_sr))
        .compose(flatten_list)
        .shuffle(1000)  # Shuffle buffer for better mixing of chunks
    )

    return dataset
