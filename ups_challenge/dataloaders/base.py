import random

import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder

from .urls import build_urls


def decode_and_normalize(
    sample,
    target_sr=16000,
    chunk_sec=10.0,
    max_chunks_per_example=16,
    shuffle_chunks=False,
):
    """
    sample comes from .to_tuple('mp3', '__key__', '__url__')
    so it's (mp3_bytes, key, url).

    We:
      - decode mp3 using torchaudio
      - resample to default_sample_rate
      - convert to mono
      - return a dict

    Any samples that fail to decode are logged and skipped.
    """
    mp3_bytes, _, _ = sample
    chunk_samples = int(chunk_sec * target_sr)

    output_chunks = []

    decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)

    duration = decoder.metadata.duration_seconds_from_header

    # ---- 2) If short file, stream entire audio ----
    if duration <= chunk_sec:
        samples = decoder.get_samples_played_in_range(0.0, duration)
        chunk = samples.data
        chunk = chunk.squeeze(0)

        # pad to exact chunk length
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        output_chunks.append(chunk)
        batch_wave = torch.stack(output_chunks)
        attention_mask = torch.ones_like(batch_wave, dtype=torch.long)
        return {
            "input_values": batch_wave,  # [N_chunks, chunk_samples]
            "attention_mask": attention_mask,  # same shape
        }

    # ---- 3) Choose random chunk start times (in seconds) ----
    max_start_sec = duration - chunk_sec

    # Generate random starting times
    start_times = [
        random.uniform(0.0, max_start_sec) for _ in range(max_chunks_per_example)
    ]

    # ---- 4) Stream each chunk ----
    for start_sec in start_times:
        end_sec = start_sec + chunk_sec

        samples = decoder.get_samples_played_in_range(start_sec, end_sec)

        chunk = samples.data
        chunk = chunk.squeeze(0)
        # Pad end-of-file short outputs
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        output_chunks.append(chunk)

    # ---- 5) Shuffle chunks across examples ----
    if shuffle_chunks:
        random.shuffle(output_chunks)

    # ---- 6) Stack into batch tensors ----
    batch_wave = torch.stack(output_chunks)

    attention_mask = torch.ones_like(batch_wave, dtype=torch.long)

    return {
        "input_values": batch_wave,  # [N_chunks, chunk_samples]
        "attention_mask": attention_mask,  # same shape
    }


def collate_fn(batch: list):
    """
    Custom collate function to:
    - handle None samples
    - concatenate input_values and attention_masks across the batch dimension
    """
    # Filter out any Nones that might slip through
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    input_values = [b["input_values"] for b in batch]

    attention_masks = [b["attention_mask"] for b in batch]
    return {
        "input_values": torch.cat(input_values, dim=0),  # (sum_N_chunks, T)
        "attention_mask": torch.cat(attention_masks, dim=0),
    }


def build_wds_dataset(langs: list = [], index_path: str = "./data/lid_index.pkl", hf_token: str = None):
    """
    Build a WebDataset dataset for the given languages.
    If langs is empty, all languages are included.
    Args:
        langs (list): List of language codes to include. If empty, all languages are included.
        index_path (str): Path to the language ID index folder.
    Returns:
        wds.WebDataset: The constructed WebDataset.
    """
    urls = build_urls(langs, index_path=index_path, hf_token=hf_token)
    return (
        wds.WebDataset(
            urls,
            shardshuffle=False,
        )
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
        .map(decode_and_normalize)
    )
