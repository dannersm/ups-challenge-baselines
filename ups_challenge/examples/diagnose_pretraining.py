"""Diagnostic script for HuBERT continued pre-training pipeline.

Checks four hypotheses for why CPT degrades performance:
  H1: Low data diversity (too many chunks from same files)
  H2: Temporal misalignment between MFCC labels and HuBERT CNN encoder
  H3: Poor k-means cluster quality
  H4: Label distribution issues

Usage:
    python -m ups_challenge.examples.diagnose_pretraining \
        --index_path ./data/pretraining_index_100h.pkl \
        --kmeans_path ./data/kmeans_100.pkl
"""

import argparse
import os
import pickle
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA


# -----------------------------------------------------------------------
# H1: Data diversity
# -----------------------------------------------------------------------

def diagnose_diversity(index):
    print("=" * 70)
    print("H1: DATA DIVERSITY")
    print("=" * 70)

    total_chunks = len(index)
    file_ids = [(e["tar_number"], e["key"]) for e in index]
    unique_files = set(file_ids)
    file_counts = Counter(file_ids)

    print(f"  Total chunks:       {total_chunks}")
    print(f"  Unique files:       {len(unique_files)}")
    print(f"  Chunks per file:    mean={total_chunks / len(unique_files):.1f}  "
          f"max={max(file_counts.values())}  min={min(file_counts.values())}")

    # Distribution of chunks per file
    counts = list(file_counts.values())
    hist_vals, bin_edges = np.histogram(counts, bins=range(1, max(counts) + 2))
    print(f"\n  Chunks-per-file distribution:")
    for i, v in enumerate(hist_vals):
        if v > 0:
            print(f"    {bin_edges[i]:3d} chunks: {v:5d} files")

    # Overlap detection within each file
    print(f"\n  Checking for overlapping chunks within each file...")
    overlap_count = 0
    overlap_total_sec = 0.0
    files_with_overlap = 0

    entries_by_file = defaultdict(list)
    for e in index:
        entries_by_file[(e["tar_number"], e["key"])].append(
            (e["start_sec"], e["end_sec"])
        )

    for fid, intervals in entries_by_file.items():
        if len(intervals) < 2:
            continue
        intervals.sort()
        file_has_overlap = False
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                overlap_sec = intervals[i][1] - intervals[i + 1][0]
                overlap_count += 1
                overlap_total_sec += overlap_sec
                file_has_overlap = True
        if file_has_overlap:
            files_with_overlap += 1

    print(f"  Overlapping chunk pairs:    {overlap_count}")
    print(f"  Files with any overlap:     {files_with_overlap} / {len(unique_files)}")
    print(f"  Total overlap time:         {overlap_total_sec:.1f} s")

    is_problem = len(unique_files) < total_chunks * 0.3
    if is_problem:
        print(f"\n  ⚠ WARNING: Only {len(unique_files)} unique files for {total_chunks} chunks.")
        print(f"    This means heavy reuse of the same audio sources.")
    else:
        print(f"\n  ✓ Diversity looks reasonable.")

    return is_problem


# -----------------------------------------------------------------------
# H2: Label distribution
# -----------------------------------------------------------------------

def diagnose_label_distribution(index, n_clusters):
    print("\n" + "=" * 70)
    print("H2: LABEL / CLUSTER DISTRIBUTION")
    print("=" * 70)

    all_labels = np.concatenate([e["labels"] for e in index])
    total_frames = len(all_labels)
    print(f"  Total label frames: {total_frames:,}")

    counts = np.bincount(all_labels, minlength=n_clusters)
    freqs = counts / total_frames

    # Entropy
    nonzero = freqs[freqs > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    max_entropy = np.log2(n_clusters)
    print(f"  Entropy:            {entropy:.3f} / {max_entropy:.3f} "
          f"({entropy / max_entropy * 100:.1f}%)")

    # Dominant / rare clusters
    rare = np.where(freqs < 0.001)[0]  # < 0.1%
    empty = np.where(counts == 0)[0]
    dominant = np.where(freqs > 0.05)[0]  # > 5%

    print(f"  Empty clusters:     {len(empty)}")
    print(f"  Rare (<0.1%):       {len(rare)} clusters")
    print(f"  Dominant (>5%):     {len(dominant)} clusters")

    if len(dominant) > 0:
        print(f"    Dominant IDs:     {dominant.tolist()}")
        for d in dominant:
            print(f"      Cluster {d}: {freqs[d]*100:.2f}%")

    # Top-10 and bottom-10
    sorted_idx = np.argsort(freqs)[::-1]
    print(f"\n  Top-10 clusters:")
    for i in sorted_idx[:10]:
        print(f"    Cluster {i:3d}: {freqs[i]*100:.2f}% ({counts[i]:,} frames)")
    print(f"  Bottom-10 clusters:")
    for i in sorted_idx[-10:]:
        print(f"    Cluster {i:3d}: {freqs[i]*100:.2f}% ({counts[i]:,} frames)")

    is_problem = entropy < max_entropy * 0.7 or len(dominant) > 5
    if is_problem:
        print(f"\n  ⚠ WARNING: Distribution is skewed (entropy {entropy:.2f} vs max {max_entropy:.2f})")
    else:
        print(f"\n  ✓ Label distribution looks reasonable.")

    return is_problem, counts, freqs


# -----------------------------------------------------------------------
# H3: K-means quality
# -----------------------------------------------------------------------

def diagnose_kmeans(kmeans_path, n_clusters):
    print("\n" + "=" * 70)
    print("H3: K-MEANS QUALITY")
    print("=" * 70)

    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    centroids = kmeans.cluster_centers_  # [K, D]
    print(f"  Centroids shape:    {centroids.shape}")
    print(f"  Inertia:            {kmeans.inertia_:.2f}")

    # Inter-cluster distances
    dists = pdist(centroids, metric="euclidean")
    print(f"  Inter-cluster distances:")
    print(f"    min:  {dists.min():.4f}")
    print(f"    mean: {dists.mean():.4f}")
    print(f"    max:  {dists.max():.4f}")
    print(f"    std:  {dists.std():.4f}")

    # Check for near-duplicate centroids
    near_dupes = np.sum(dists < dists.mean() * 0.1)
    print(f"  Near-duplicate pairs (<10% of mean dist): {near_dupes}")

    # Centroid norms (detect if unnormalized MFCCs cause scale issues)
    norms = np.linalg.norm(centroids, axis=1)
    print(f"\n  Centroid L2 norms:")
    print(f"    min:  {norms.min():.2f}")
    print(f"    mean: {norms.mean():.2f}")
    print(f"    max:  {norms.max():.2f}")
    print(f"    ratio max/min: {norms.max() / norms.min():.2f}")

    # Feature-wise stats of centroids
    print(f"\n  Per-feature centroid stats (first 13 = static MFCC):")
    for i in range(min(13, centroids.shape[1])):
        col = centroids[:, i]
        print(f"    dim {i:2d}: mean={col.mean():8.2f}  std={col.std():7.2f}  "
              f"range=[{col.min():.2f}, {col.max():.2f}]")

    is_problem = norms.max() / norms.min() > 10 or near_dupes > 5
    if is_problem:
        print(f"\n  ⚠ WARNING: K-means may have quality issues (norm ratio or near-dupes).")
    else:
        print(f"\n  ✓ K-means quality looks reasonable.")

    return is_problem, kmeans, centroids


# -----------------------------------------------------------------------
# H4: Temporal alignment (offline check, no audio needed)
# -----------------------------------------------------------------------

def diagnose_alignment(index):
    print("\n" + "=" * 70)
    print("H4: TEMPORAL ALIGNMENT (offline estimation)")
    print("=" * 70)

    # For each chunk, we know the audio duration (end_sec - start_sec)
    # and the number of label frames. We can estimate what the CNN encoder
    # would produce and compare.

    # HuBERT CNN: effective stride = 320 samples at 16kHz = 20ms
    # But the 7-layer CNN has specific kernel sizes that affect the output length.
    # For a waveform of N samples, output length ≈ floor((N - 400) / 320) + 1
    # (this is an approximation; exact formula depends on all 7 conv layers)

    # MFCC with hop_length=320, n_fft=400:
    # output length = floor((N - 400) / 320) + 1  (centered) or floor(N / 320) (not centered)
    # torchaudio MFCC uses centered=True by default, so:
    # mfcc_len ≈ floor(N / 320) + 1

    sample_rate = 16000
    hop_length = 320

    deltas = []
    label_lens = []
    estimated_cnn_lens = []
    estimated_mfcc_lens = []

    for entry in index[:1000]:  # check first 1000 entries
        duration = entry["end_sec"] - entry["start_sec"]
        n_samples = int(duration * sample_rate)
        n_labels = len(entry["labels"])

        # MFCC length (torchaudio centered=True): ceil(n_samples / hop_length)
        # Actually: n_fft=400, so with padding: floor(n_samples / hop_length) + 1
        est_mfcc = n_samples // hop_length + 1

        # HuBERT CNN encoder output length (empirical formula for hubert-base):
        # After 7 conv layers with strides [5,2,2,2,2,2,2] = product 640... wait
        # Actually HuBERT uses the wav2vec2 feature extractor:
        # conv layers: [(512,10,5), (512,3,2), (512,3,2), (512,3,2), (512,3,2), (512,2,2), (512,2,2)]
        # Total stride = 5*2*2*2*2*2*2 = 320
        # Receptive field = 400 samples
        # Output length = floor((n_samples - 400) / 320) + 1  (no padding in wav2vec2 CNN)
        est_cnn = max(0, (n_samples - 400) // 320 + 1)

        delta = n_labels - est_cnn
        deltas.append(delta)
        label_lens.append(n_labels)
        estimated_cnn_lens.append(est_cnn)
        estimated_mfcc_lens.append(est_mfcc)

    deltas = np.array(deltas)
    label_lens = np.array(label_lens)
    estimated_cnn_lens = np.array(estimated_cnn_lens)
    estimated_mfcc_lens = np.array(estimated_mfcc_lens)

    print(f"  Checked {len(deltas)} entries")
    print(f"\n  Label frames (from MFCC):")
    print(f"    mean: {label_lens.mean():.1f}  std: {label_lens.std():.1f}")
    print(f"  Estimated CNN encoder output frames:")
    print(f"    mean: {estimated_cnn_lens.mean():.1f}  std: {estimated_cnn_lens.std():.1f}")
    print(f"  Estimated MFCC frames (centered):")
    print(f"    mean: {estimated_mfcc_lens.mean():.1f}  std: {estimated_mfcc_lens.std():.1f}")

    print(f"\n  Delta (n_labels - est_cnn_len):")
    print(f"    mean:   {deltas.mean():.2f}")
    print(f"    std:    {deltas.std():.2f}")
    print(f"    min:    {deltas.min()}")
    print(f"    max:    {deltas.max()}")
    print(f"    median: {np.median(deltas):.0f}")

    # Check if there's a systematic offset
    unique_deltas, delta_counts = np.unique(deltas, return_counts=True)
    print(f"\n  Delta distribution:")
    for d, c in zip(unique_deltas, delta_counts):
        print(f"    delta={d:+3d}: {c:5d} entries ({c/len(deltas)*100:.1f}%)")

    # The key question: does the training code's min() truncation cause misalignment?
    mfcc_vs_cnn_diff = estimated_mfcc_lens - estimated_cnn_lens
    print(f"\n  MFCC len vs CNN len difference (est_mfcc - est_cnn):")
    print(f"    mean:   {mfcc_vs_cnn_diff.mean():.2f}")
    print(f"    This means MFCC frames are systematically "
          f"{'longer' if mfcc_vs_cnn_diff.mean() > 0 else 'shorter'} than CNN output.")
    print(f"    The training loop clips to min(enc_len, label_len), which may drop "
          f"{abs(mfcc_vs_cnn_diff.mean()):.0f} trailing frames per sample.")

    is_problem = abs(deltas.mean()) > 1.5 or deltas.std() > 2.0
    if is_problem:
        print(f"\n  ⚠ WARNING: Systematic offset detected. Labels and encoder may be misaligned.")
    else:
        print(f"\n  ✓ Alignment looks reasonable (small/consistent offset).")

    return is_problem


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def save_plots(index, counts, freqs, centroids, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Chunks per file histogram
    file_ids = [(e["tar_number"], e["key"]) for e in index]
    file_counts = Counter(file_ids)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(file_counts.values(), bins=range(1, max(file_counts.values()) + 2),
            edgecolor="black", alpha=0.7)
    ax.set_xlabel("Chunks per file")
    ax.set_ylabel("Number of files")
    ax.set_title("Distribution of chunks per source file")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "chunks_per_file_hist.png"), dpi=150)
    plt.close(fig)

    # 2. Cluster distribution
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(counts)), freqs * 100, color="steelblue", edgecolor="none")
    ax.axhline(y=100 / len(counts), color="red", linestyle="--", alpha=0.5,
               label=f"Uniform = {100/len(counts):.2f}%")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("K-means cluster distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cluster_distribution.png"), dpi=150)
    plt.close(fig)

    # 3. Centroids PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(centroids)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=freqs * 100, cmap="viridis",
                         s=50, edgecolors="black", linewidth=0.5)
    for i in range(len(centroids)):
        ax.annotate(str(i), (proj[i, 0], proj[i, 1]), fontsize=5, alpha=0.6)
    plt.colorbar(scatter, label="Cluster freq (%)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("K-means centroids (PCA)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "centroids_pca.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Plots saved to {output_dir}/")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose HuBERT pretraining pipeline")
    parser.add_argument("--index_path", type=str,
                        default="./data/pretraining_index_100h.pkl")
    parser.add_argument("--kmeans_path", type=str,
                        default="./data/kmeans_100.pkl")
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--plot_dir", type=str, default="./data/diagnostics")
    args = parser.parse_args()

    print("Loading index...")
    with open(args.index_path, "rb") as f:
        index = pickle.load(f)
    print(f"  Loaded {len(index)} entries\n")

    problems = []

    # H1
    p1 = diagnose_diversity(index)
    problems.append(("H1: Data diversity", p1))

    # H2 (label distribution)
    p2, counts, freqs = diagnose_label_distribution(index, args.n_clusters)
    problems.append(("H2: Label distribution", p2))

    # H3
    p3, kmeans, centroids = diagnose_kmeans(args.kmeans_path, args.n_clusters)
    problems.append(("H3: K-means quality", p3))

    # H4
    p4 = diagnose_alignment(index)
    problems.append(("H4: Temporal alignment", p4))

    # Plots
    save_plots(index, counts, freqs, centroids, args.plot_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    any_problem = False
    for name, is_problem in problems:
        status = "⚠ PROBLEM" if is_problem else "✓ OK"
        print(f"  {status:12s}  {name}")
        if is_problem:
            any_problem = True

    if any_problem:
        print("\n  One or more issues detected. Review the details above.")
        return 1
    else:
        print("\n  All checks passed.")
        return 0


if __name__ == "__main__":
    exit(main())
