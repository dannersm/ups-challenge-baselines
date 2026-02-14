
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
