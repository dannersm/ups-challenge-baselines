"""Offline script to build a pretraining index with pre-computed k-means labels.

Single-pass approach:
  1. Stream audio via WebDataset, extract random 10s chunks, compute 39-dim MFCC.
  2. Fit MiniBatchKMeans on collected MFCC frames.
  3. Assign k-means cluster IDs as labels to all chunks.
  4. Save pretraining index (.pkl) + k-means model (.pkl).

Usage:
    python -m ups_challenge.examples.prepare_pretraining_index \
        --hours 100 --n_clusters 100 --output_dir ./data

Note: buffers all MFCCs in RAM (~2.8 GB for 100 h, ~28 GB for 1000 h).
For very large subsets, a two-pass approach would be needed.
"""

import argparse
import os
import pickle
import random
import re

import numpy as np
import torch
import torchaudio
import webdataset as wds
from sklearn.cluster import MiniBatchKMeans
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sequential_urls(hf_token, max_shards=2000):
    """Build pipe:curl URLs for sequential tars (000000.tar to max_shards)."""
    token = f"Authorization:Bearer {hf_token}"
    urls = []
    for i in range(max_shards):
        tn = f"{i:06d}"
        folder = "audio" if i <= 5000 else "audio2"
        raw = (
            f"https://huggingface.co/datasets/MLCommons/"
            f"unsupervised_peoples_speech/resolve/main/{folder}/{tn}.tar?download=True"
        )
        urls.append(f"pipe:curl -s -L {raw} -H {token}")
    return urls


def extract_mfcc(waveform, sample_rate=16000, n_mfcc=13, hop_length=320):
    """Extract 39-dim MFCC (13 static + delta + delta-delta).

    hop_length=320 at 16 kHz -> 20 ms step -> 50 Hz, matching HuBERT output rate.
    Returns np.ndarray of shape [frames, 39].
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": hop_length, "n_mels": 23},
    )

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mfcc = mfcc_transform(waveform)                     # [1, n_mfcc, frames]
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    features = torch.cat([mfcc, delta, delta2], dim=1)   # [1, 39, frames]

    return features.squeeze(0).T.numpy()                 # [frames, 39]


def _decode_chunk(sample, target_sr=16000, chunk_sec=10.0, max_chunks_per_file=16):
    """Decode one MP3 -> multiple random 10 s chunks. Returns list of dicts."""
    mp3_bytes, key, url = sample

    try:
        decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)
        duration = decoder.metadata.duration_seconds_from_header
    except Exception:
        return []

    if duration < chunk_sec:
        return []

    # Extract tar number from URL  ("…/audio/000123.tar?…")
    m = re.search(r"/(\d+)\.tar", url)
    tar_number = m.group(1) if m else "unknown"

    # Strategy: take up to N chunks from this file
    max_start = max(0.0, duration - chunk_sec)
    
    # If the file is long enough, we can take multiple non-overlapping or random chunks
    # Simple approach: random starts
    current_chunks = []
    for _ in range(max_chunks_per_file):
        start_sec = random.uniform(0.0, max_start)
        end_sec = start_sec + chunk_sec

        try:
            waveform = decoder.get_samples_played_in_range(start_sec, end_sec).data.squeeze(0)
            # Verify length (sometimes slight mismatch due to seeking precision)
            if waveform.shape[0] > 0:
                current_chunks.append({
                    "waveform": waveform,
                    "key": key,
                    "tar_number": tar_number,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                })
        except Exception:
            continue
            
    return current_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build pretraining index with k-means labels"
    )
    parser.add_argument("--hours", type=float, default=100.0,
                        help="Target hours of audio")
    parser.add_argument("--n_clusters", type=int, default=100,
                        help="Number of k-means clusters")
    parser.add_argument("--chunk_sec", type=float, default=10.0,
                        help="Chunk duration in seconds")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for index and k-means model")
    parser.add_argument("--index_path", type=str, default="./data/lid_index.pkl",
                        help="Path to the lid_index (used to discover tar shards)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--kmeans_sample_frames", type=int, default=500_000,
                        help="Max MFCC frames to subsample for k-means fitting")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("Set HF_TOKEN env var or pass --hf_token")

    target_chunks = int(args.hours * 3600 / args.chunk_sec)
    print(f"Target: {args.hours} h = {target_chunks} chunks of {args.chunk_sec} s")

    # Generate sequential URLs (enough to cover the target)
    urls = _build_sequential_urls(hf_token, max_shards=2000)
    print(f"Streaming sequentially from up to {len(urls)} tar shards...")

    # ------------------------------------------------------------------
    # Single pass: collect chunks + MFCC
    # ------------------------------------------------------------------
    dataset = (
        wds.WebDataset(urls, shardshuffle=False, handler=wds.handlers.ignore_and_continue)
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
        .map(lambda s: _decode_chunk(s, chunk_sec=args.chunk_sec))
    )

    chunks = []       # metadata dicts (no waveform)
    mfcc_list = []    # parallel list of MFCC arrays
    total_frames = 0

    pbar = tqdm(total=target_chunks, desc="Collecting chunks")
    
    # Dataset yields lists of chunks now
    for batch_chunks in dataset:
        if not batch_chunks:
            continue

        for item in batch_chunks:
            waveform = item.pop("waveform")
            mfcc = extract_mfcc(waveform)

            chunks.append(item)
            mfcc_list.append(mfcc)
            total_frames += mfcc.shape[0]
            pbar.update(1)

        if len(chunks) >= target_chunks:
            break
    pbar.close()

    print(f"Collected {len(chunks)} chunks ({total_frames} MFCC frames)")

    # ------------------------------------------------------------------
    # Fit k-means
    # ------------------------------------------------------------------
    all_frames = np.concatenate(mfcc_list, axis=0)
    if all_frames.shape[0] > args.kmeans_sample_frames:
        idx = np.random.choice(all_frames.shape[0], args.kmeans_sample_frames, replace=False)
        fit_frames = all_frames[idx]
    else:
        fit_frames = all_frames
    del all_frames

    print(f"Fitting MiniBatchKMeans(K={args.n_clusters}) on {fit_frames.shape[0]} frames ...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size=1024,
        random_state=42,
        n_init=3,
    )
    kmeans.fit(fit_frames)
    del fit_frames
    print(f"Inertia: {kmeans.inertia_:.2f}")

    # ------------------------------------------------------------------
    # Assign labels
    # ------------------------------------------------------------------
    print("Assigning k-means labels ...")
    index_entries = []
    for chunk, mfcc in tqdm(zip(chunks, mfcc_list), total=len(chunks), desc="Labeling"):
        labels = kmeans.predict(mfcc).astype(np.int16)
        index_entries.append({
            "tar_number": chunk["tar_number"],
            "key": chunk["key"],
            "start_sec": chunk["start_sec"],
            "end_sec": chunk["end_sec"],
            "labels": labels,
        })
    del mfcc_list

    # Shuffle the final index so training batches aren't sequential by shard
    print("Shuffling index entries...")
    random.shuffle(index_entries)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    idx_path = os.path.join(args.output_dir, f"pretraining_index_{int(args.hours)}h.pkl")
    with open(idx_path, "wb") as f:
        pickle.dump(index_entries, f, protocol=4)
    print(f"Saved index -> {idx_path} ({len(index_entries)} entries)")

    km_path = os.path.join(args.output_dir, f"kmeans_{args.n_clusters}.pkl")
    with open(km_path, "wb") as f:
        pickle.dump(kmeans, f, protocol=4)
    print(f"Saved k-means -> {km_path}")


if __name__ == "__main__":
    main()