"""Offline script to build a pretraining index with pre-computed k-means labels.

Approach:
  1. Stream tars from HuggingFace via WebDataset.
  2. For each audio file, look up its VAD segments and cut deterministic
     non-overlapping 10 s chunks anchored to speech segment ends.
  3. Extract 39-dim MFCCs (13 + delta + delta-delta) with center=False so
     frame count matches HuBERT's CNN output (499 frames per 10 s chunk).
  4. Z-score normalise MFCCs globally before fitting k-means.
  5. Fit MiniBatchKMeans on a random subsample of frames.
  6. Assign cluster IDs as labels and save index + k-means model.

Usage:
    python -m ups_challenge.examples.prepare_pretraining_index \\
        --hours 100 --n_clusters 100 --output_dir ./data \\
        --vad_base_dir ./data/vad_shards \\
        --lid_index_path ./data/lid_index.pkl

Prerequisites:
  - VAD shards: python -m ups_challenge.vad_analysis.vad_lookup
  - lid_index.pkl (only used to know which tars have any speech at all)

Note: buffers all MFCCs in RAM (~2.8 GB for 100 h).
"""

import argparse
import os
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio
import webdataset as wds
from sklearn.cluster import MiniBatchKMeans
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from ups_challenge.vad_analysis.apply_vad import (
    _parse_tar_number_from_url,
    load_vad_shard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vad_chunks_for_file(segments, chunk_sec=10.0):
    """Non-overlapping fixed-length chunks anchored to speech segment ends."""
    chunks = []
    for seg_start, seg_end in segments:
        if seg_end - seg_start < chunk_sec:
            continue
        anchor = seg_end
        while anchor - seg_start >= chunk_sec:
            chunks.append((anchor - chunk_sec, anchor))
            anchor -= chunk_sec
    return chunks


def extract_mfcc(waveform, sample_rate=16000, n_mfcc=13, hop_length=320):
    """39-dim MFCC (13 + delta + delta²), center=False → 499 frames per 10 s."""
    mfcc_t = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": hop_length, "n_mels": 23, "center": False},
    )
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mfcc = mfcc_t(waveform)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    return torch.cat([mfcc, delta, delta2], dim=1).squeeze(0).T.numpy()


def _build_tar_urls(tar_numbers, hf_token):
    token = f"Authorization:Bearer {hf_token}"
    urls = []
    for tn in sorted(tar_numbers):
        folder = "audio" if int(tn) <= 5000 else "audio2"
        raw = (
            f"https://huggingface.co/datasets/MLCommons/"
            f"unsupervised_peoples_speech/resolve/main/{folder}/{tn}.tar?download=True"
        )
        urls.append(f"pipe:curl -s -L {raw} -H {token}")
    return urls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build pretraining index: VAD chunks → MFCCs → k-means labels"
    )
    parser.add_argument("--hours", type=float, default=100.0,
                        help="Target hours of audio to collect")
    parser.add_argument("--n_clusters", type=int, default=100,
                        help="Number of k-means clusters")
    parser.add_argument("--chunk_sec", type=float, default=10.0,
                        help="Chunk duration in seconds")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--kmeans_sample_frames", type=int, default=500_000,
                        help="Max MFCC frames subsampled for k-means fitting")
    parser.add_argument("--vad_base_dir", type=str, default="./data/vad_shards")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Set HF_TOKEN env var or pass --hf_token")

    target_chunks = int(args.hours * 3600 / args.chunk_sec)
    print(f"Target: {args.hours} h = {target_chunks} chunks of {args.chunk_sec} s")

    # ------------------------------------------------------------------
    # Find tars that have VAD shards
    # ------------------------------------------------------------------
    vad_base = Path(args.vad_base_dir)
    if not vad_base.exists() or not list(vad_base.glob("*.pkl")):
        raise FileNotFoundError(
            f"VAD shards not found in {vad_base}. "
            "Run: python -m ups_challenge.vad_analysis.vad_lookup"
        )

    available_tars = list({f.stem for f in vad_base.glob("*.pkl")})
    random.shuffle(available_tars)
    print(f"Available VAD shards: {len(available_tars)} tars (shuffled)")

    urls = _build_tar_urls(available_tars, hf_token)
    print(f"Streaming from {len(urls)} tars...")

    # ------------------------------------------------------------------
    # Single pass: stream → VAD chunks → MFCC
    # ------------------------------------------------------------------
    dataset = (
        wds.WebDataset(urls, shardshuffle=False, handler=wds.handlers.ignore_and_continue)
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
    )

    chunks = []
    mfcc_list = []
    total_frames = 0

    pbar = tqdm(total=target_chunks, desc="Collecting chunks")

    for mp3_bytes, key, url in dataset:
        if len(chunks) >= target_chunks:
            break

        tar_number = _parse_tar_number_from_url(url)
        if tar_number is None:
            continue

        filename = os.path.basename(key)
        if not filename.endswith(".mp3"):
            filename += ".mp3"

        try:
            vad_shard = load_vad_shard(tar_number, base_dir=args.vad_base_dir)
        except FileNotFoundError:
            continue
        vad_data = vad_shard.get(filename)
        if vad_data is None or not vad_data.get("segments"):
            continue

        file_chunks = vad_chunks_for_file(vad_data["segments"], chunk_sec=args.chunk_sec)
        if not file_chunks:
            continue

        # Don't overshoot target
        remaining = target_chunks - len(chunks)
        file_chunks = file_chunks[:remaining]

        try:
            decoder = AudioDecoder(source=mp3_bytes, sample_rate=16000, num_channels=1)
        except Exception:
            continue

        for start_sec, end_sec in file_chunks:
            try:
                waveform = decoder.get_samples_played_in_range(
                    start_sec, end_sec
                ).data.squeeze(0)
                if waveform.shape[0] == 0:
                    continue
            except Exception:
                continue

            mfcc = extract_mfcc(waveform)
            chunks.append({
                "tar_number": tar_number,
                "key": key,
                "start_sec": start_sec,
                "end_sec": end_sec,
            })
            mfcc_list.append(mfcc)
            total_frames += mfcc.shape[0]
            pbar.update(1)

        if len(chunks) >= target_chunks:
            break

    pbar.close()

    print(f"\nCollected {len(chunks)} chunks ({total_frames} MFCC frames)")
    if not chunks:
        print("ERROR: No chunks collected. Check VAD shards.")
        return

    # ------------------------------------------------------------------
    # Global z-score normalisation
    # ------------------------------------------------------------------
    print("Computing global MFCC normalisation...")
    all_frames = np.concatenate(mfcc_list, axis=0)
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)
    print(f"  Mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  Std  range: [{std.min():.4f}, {std.max():.4f}]")

    print("Normalising MFCCs...")
    for i in range(len(mfcc_list)):
        mfcc_list[i] = (mfcc_list[i] - mean) / (std + 1e-8)

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

    print(f"Fitting MiniBatchKMeans(K={args.n_clusters}) on {fit_frames.shape[0]} frames...")
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, batch_size=1024,
                             random_state=42, n_init=3)
    kmeans.fit(fit_frames)
    del fit_frames
    print(f"Inertia: {kmeans.inertia_:.2f}")

    norms = np.linalg.norm(kmeans.cluster_centers_, axis=1)
    print(f"Centroid norm ratio max/min: {norms.max() / (norms.min() + 1e-8):.2f}x "
          f"(should be <3x with normalisation)")

    # ------------------------------------------------------------------
    # Assign labels
    # ------------------------------------------------------------------
    print("Assigning k-means labels...")
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

    random.shuffle(index_entries)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    idx_path = os.path.join(args.output_dir, f"pretraining_index_{int(args.hours)}h.pkl")
    with open(idx_path, "wb") as f:
        pickle.dump(index_entries, f, protocol=4)
    print(f"Saved index  → {idx_path}  ({len(index_entries)} entries)")

    km_path = os.path.join(args.output_dir, f"kmeans_{args.n_clusters}.pkl")
    with open(km_path, "wb") as f:
        pickle.dump(kmeans, f, protocol=4)
    print(f"Saved kmeans → {km_path}")

    norm_path = os.path.join(args.output_dir, f"kmeans_norm_{args.n_clusters}.pkl")
    with open(norm_path, "wb") as f:
        pickle.dump({"mean": mean, "std": std, "kmeans": kmeans}, f, protocol=4)
    print(f"Saved kmeans + norm stats → {norm_path}")

    # Summary
    unique_files = len({(e["tar_number"], e["key"]) for e in index_entries})
    unique_tars = len({e["tar_number"] for e in index_entries})
    label_lens = [len(e["labels"]) for e in index_entries]
    print(f"\nFinal summary:")
    print(f"  Entries:      {len(index_entries)}")
    print(f"  Unique files: {unique_files}")
    print(f"  Unique tars:  {unique_tars}")
    print(f"  Label lengths: min={min(label_lens)}, max={max(label_lens)}, "
          f"mode={Counter(label_lens).most_common(1)[0]}")


if __name__ == "__main__":
    main()
