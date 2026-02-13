"""Phase 2: Stream tars sequentially, extract chunks, compute MFCCs, fit k-means.

Reads the chunk index produced by build_chunk_index.py (sorted by tar_number),
streams tars one at a time via WebDataset, decodes audio chunks, computes
39-dim MFCCs, fits MiniBatchKMeans, and saves the labeled index.

Usage:
    python -m ups_challenge.examples.assign_labels \\
        --index ./data/chunk_index_100h.pkl \\
        --n_clusters 100 --output_dir ./data \\
        --hf_token $HF_TOKEN

Output files:
    data/pretraining_index_100h.pkl   -- chunk entries + 'labels' field
    data/kmeans_100.pkl               -- fitted k-means model
    data/kmeans_norm_100.pkl          -- {mean, std, kmeans} for reproducibility
"""

import argparse
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
import webdataset as wds
from sklearn.cluster import MiniBatchKMeans
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Audio / MFCC
# ---------------------------------------------------------------------------

def extract_mfcc(waveform, sample_rate=16000, n_mfcc=13, hop_length=320):
    """39-dim MFCC (13 + delta + delta²), center=False → 499 frames per 10 s."""
    mfcc_t = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": hop_length,
                   "n_mels": 23, "center": False},
    )
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mfcc = mfcc_t(waveform)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    return torch.cat([mfcc, delta, delta2], dim=1).squeeze(0).T.numpy()


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

def _build_tar_url(tar_number: str, hf_token: str) -> str:
    token = f"Authorization:Bearer {hf_token}"
    tn_str = str(tar_number).zfill(6)
    folder = "audio" if int(tar_number) <= 5000 else "audio2"
    raw = (
        f"https://huggingface.co/datasets/MLCommons/"
        f"unsupervised_peoples_speech/resolve/main/{folder}/{tn_str}.tar?download=True"
    )
    return f"pipe:curl -s -L {raw} -H {token}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Assign k-means labels to a Phase 1 chunk index"
    )
    parser.add_argument("--index", type=str, required=True,
                        help="Path to chunk index produced by build_chunk_index.py")
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache downloaded tars (reused on subsequent runs)")
    parser.add_argument("--kmeans_sample_frames", type=int, default=500_000,
                        help="Max MFCC frames subsampled for k-means fitting")
    parser.add_argument("--target_sr", type=int, default=16000)
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Set HF_TOKEN env var or pass --hf_token")

    # ------------------------------------------------------------------
    # Load index
    # ------------------------------------------------------------------
    print(f"Loading index from {args.index}...")
    with open(args.index, "rb") as f:
        index_entries = pickle.load(f)
    print(f"  {len(index_entries):,} entries")

    # Build lookup: (tar_number, key) -> list[entry_index]
    lookup: dict[tuple, list[int]] = defaultdict(list)
    for i, entry in enumerate(index_entries):
        lookup[(entry["tar_number"], entry["key"])].append(i)

    # Get ordered list of unique tars (index is already tar-sorted)
    seen = set()
    ordered_tars = []
    for entry in index_entries:
        t = entry["tar_number"]
        if t not in seen:
            ordered_tars.append(t)
            seen.add(t)
    print(f"  {len(ordered_tars)} unique tars, streaming sequentially")

    # ------------------------------------------------------------------
    # Stream tars, decode chunks, compute MFCCs
    # ------------------------------------------------------------------
    mfcc_list = [None] * len(index_entries)  # pre-allocated by entry index
    collected_count = 0

    pbar = tqdm(total=len(index_entries), desc="Chunks decoded", unit="chunk")

    for tar_idx, tar_number in enumerate(ordered_tars):
        import time
        chunks_in_tar = sum(1 for e in index_entries if e["tar_number"] == tar_number)
        t0 = time.time()
        first_chunk_logged = False
        url = _build_tar_url(tar_number, hf_token)
        pbar.write(f"\n[{tar_idx+1}/{len(ordered_tars)}] Connecting to tar {tar_number} "
                   f"({chunks_in_tar:,} chunks expected) ...")

        # Build per-tar lookup
        tar_lookup: dict[str, list[int]] = defaultdict(list)
        for (tn, key), idxs in lookup.items():
            if tn == tar_number:
                tar_lookup[key].extend(idxs)

        if not tar_lookup:
            continue

        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        dataset = (
            wds.WebDataset(url, shardshuffle=False,
                           cache_dir=str(cache_dir) if cache_dir else None,
                           handler=wds.handlers.ignore_and_continue)
            .to_tuple("mp3", "__key__", "__url__",
                      handler=wds.handlers.ignore_and_continue)
        )

        for mp3_bytes, key, _url in dataset:
            # WDS key is full path in tar (e.g. "audio/00001"), index stores basename only
            idxs = tar_lookup.get(os.path.basename(key))
            if idxs is None:
                continue

            try:
                decoder = AudioDecoder(source=mp3_bytes,
                                       sample_rate=args.target_sr,
                                       num_channels=1)
            except Exception:
                continue

            for idx in idxs:
                entry = index_entries[idx]
                try:
                    waveform = decoder.get_samples_played_in_range(
                        entry["start_sec"], entry["end_sec"]
                    ).data.squeeze(0)
                    if waveform.shape[0] == 0:
                        continue
                except Exception:
                    continue

                mfcc = extract_mfcc(waveform, sample_rate=args.target_sr)
                mfcc_list[idx] = mfcc
                collected_count += 1
                if not first_chunk_logged:
                    pbar.write(f"  First chunk arrived after {time.time()-t0:.1f}s")
                    first_chunk_logged = True
                pbar.update(1)

    pbar.close()
    print(f"\nDecoded {collected_count:,} / {len(index_entries):,} chunks")

    # Drop entries where MFCC extraction failed
    valid_pairs = [(i, mfcc_list[i]) for i in range(len(index_entries))
                   if mfcc_list[i] is not None]
    if not valid_pairs:
        print("ERROR: No MFCCs collected.")
        return

    valid_idxs = [p[0] for p in valid_pairs]
    valid_mfccs = [p[1] for p in valid_pairs]
    print(f"Valid chunks for labeling: {len(valid_pairs):,}")

    # ------------------------------------------------------------------
    # Global z-score normalisation
    # ------------------------------------------------------------------
    print("Computing global MFCC normalisation...")
    all_frames = np.concatenate(valid_mfccs, axis=0)
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)
    print(f"  Mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  Std  range: [{std.min():.4f}, {std.max():.4f}]")

    print("Normalising MFCCs...")
    for i in range(len(valid_mfccs)):
        valid_mfccs[i] = (valid_mfccs[i] - mean) / (std + 1e-8)

    # ------------------------------------------------------------------
    # Fit k-means
    # ------------------------------------------------------------------
    all_frames = np.concatenate(valid_mfccs, axis=0)
    if all_frames.shape[0] > args.kmeans_sample_frames:
        idx = np.random.choice(all_frames.shape[0], args.kmeans_sample_frames,
                               replace=False)
        fit_frames = all_frames[idx]
    else:
        fit_frames = all_frames
    del all_frames

    print(f"Fitting MiniBatchKMeans(K={args.n_clusters}) "
          f"on {fit_frames.shape[0]:,} frames...")
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
    for i, mfcc in tqdm(zip(valid_idxs, valid_mfccs),
                         total=len(valid_idxs), desc="Labeling"):
        index_entries[i]["labels"] = kmeans.predict(mfcc).astype(np.int16)

    # Only keep entries that have labels
    labeled_entries = [index_entries[i] for i in valid_idxs]
    print(f"Labeled entries: {len(labeled_entries):,}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # Infer hours from index name or count
    n_hours = len(labeled_entries) * 10 / 3600
    suffix = f"{round(n_hours)}h"

    idx_path = os.path.join(args.output_dir, f"pretraining_index_{suffix}.pkl")
    with open(idx_path, "wb") as f:
        pickle.dump(labeled_entries, f, protocol=4)
    print(f"Saved index  → {idx_path}  ({len(labeled_entries):,} entries)")

    km_path = os.path.join(args.output_dir, f"kmeans_{args.n_clusters}.pkl")
    with open(km_path, "wb") as f:
        pickle.dump(kmeans, f, protocol=4)
    print(f"Saved kmeans → {km_path}")

    norm_path = os.path.join(args.output_dir,
                             f"kmeans_norm_{args.n_clusters}.pkl")
    with open(norm_path, "wb") as f:
        pickle.dump({"mean": mean, "std": std, "kmeans": kmeans}, f, protocol=4)
    print(f"Saved kmeans + norm stats → {norm_path}")

    # Summary
    unique_files = len({(e["tar_number"], e["key"]) for e in labeled_entries})
    unique_tars = len({e["tar_number"] for e in labeled_entries})
    label_lens = [len(e["labels"]) for e in labeled_entries]
    lang_counts = Counter(e.get("language", "unknown") for e in labeled_entries)
    print(f"\nFinal summary:")
    print(f"  Entries:      {len(labeled_entries):,}")
    print(f"  Unique files: {unique_files:,}")
    print(f"  Unique tars:  {unique_tars}")
    print(f"  Label lengths: min={min(label_lens)}, max={max(label_lens)}, "
          f"mode={Counter(label_lens).most_common(1)[0]}")
    print(f"  Languages ({len(lang_counts)}):")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:6s}  {cnt:6,} chunks")


if __name__ == "__main__":
    main()
