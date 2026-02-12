import torch
import webdataset as wds
import pickle 
from collections import Counter
from webdataset import RandomMix

from ups_challenge.dataloaders.urls import build_urls
from ups_challenge.vad_analysis.apply_vad import decode_and_normalize_with_vad, collate_fn


if __name__ == "__main__":
    print(">>> ENTERED __main__")

    langs = ["es", "ar", "fr"]
    batch_size = 1 
    num_workers = 0
    max_batches = 20

    index_path = "./data/lid_index.pkl"
    with open(index_path, "rb") as f:
        lid_index = pickle.load(f)
    datasets = []

    for lang in langs:
        urls = build_urls(langs=[lang])
        print(f"[{lang.upper()}] Found {len(urls)} tar files")

        ds = (
            wds.WebDataset(urls, shardshuffle=1000)
            .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
            .map(lambda x: decode_and_normalize_with_vad(
                x, desired_languages=langs, tar_to_lang=lid_index)))

        datasets.append(ds)
    
    print("\n>>> Building interleaved WebDataset with RandomMix")
    wds_dataset = RandomMix(datasets, longest=False)

    data_loader = torch.utils.data.DataLoader(
        wds_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    total_chunks = 0
    batch_count = 0
    
    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break

        if batch is None:
            print(f"[WARNING] Got None batch at iteration {i}, continuing...")
            continue

        waves = batch["input_values"]
        lang_dist = Counter(batch["language"])

        total_chunks += waves.shape[0]
        batch_count += 1

        print(f"\n[BATCH {i}] Got {waves.shape[0]} chunks")
        print(f"[BATCH {i}] Language distribution: {dict(lang_dist)}")
